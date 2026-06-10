import os
import time
import atexit
import shutil
import threading
import multiprocessing as mp
from queue import Empty
from multiprocessing.connection import Listener, Client

import click
import yaml
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer
from safetensors.torch import load_file, save_file

torch.set_printoptions(linewidth=200)


# =========================
# target feature server/client
# =========================
def _target_feat_server(
    target_model_name: str,
    cache_dir: str,
    address: tuple,
    device: str,
    authkey: bytes,
    target_layers: list,
):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(1)

    model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    lock = threading.Lock()
    listener = Listener(address, authkey=authkey)

    def handle_conn(conn):
        try:
            while True:
                msg = conn.recv()
                if msg is None or msg.get("cmd") == "quit":
                    break

                input_ids = msg["input_ids"].to(device, non_blocking=True)
                attention_mask = msg.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device, non_blocking=True)

                with lock, torch.inference_mode():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                    last_hidden = outputs.hidden_states[-1]  # [B, T, H]
                    last_hidden = torch.concat([outputs["hidden_states"][i] for i in target_layers], axis=-1)


                    target_feat = torch.cat(
                        [torch.zeros_like(last_hidden[:, :1]), last_hidden[:, :-1]],
                        dim=1,
                    ).cpu()

                    target_logits = outputs.hidden_states[-1].cpu()

                    teacher_gold_prob = torch.ones(
                        input_ids.shape,
                        dtype=torch.float32,
                        device=device,
                    )

                    if input_ids.shape[1] > 1:
                        shift_logits = outputs.logits[:, :-1, :].float()      # [B, T-1, V]
                        shift_labels = input_ids[:, 1:].unsqueeze(-1)         # [B, T-1, 1]

                        gold_log_prob = F.log_softmax(shift_logits, dim=-1).gather(
                            dim=-1,
                            index=shift_labels,
                        ).squeeze(-1)                                          # [B, T-1]

                        teacher_gold_prob[:, 1:] = gold_log_prob.exp()

                    teacher_gold_prob = teacher_gold_prob.cpu()

                conn.send(
                    {
                        "target_feat": target_feat,
                        "target_logits": target_logits,
                        "teacher_gold_prob": teacher_gold_prob,
                    }
                )
        except EOFError:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    try:
        while True:
            conn = listener.accept()
            t = threading.Thread(target=handle_conn, args=(conn,), daemon=True)
            t.start()
    finally:
        listener.close()


class TargetFeatClient:

    def __init__(self, address: tuple, authkey: bytes, timeout_s: float = 30.0):
        self.address = address
        self.authkey = authkey
        self.timeout_s = timeout_s
        self._conn = None

    def _connect(self):
        if self._conn is not None:
            return

        t0 = time.time()
        while True:
            try:
                self._conn = Client(self.address, authkey=self.authkey)
                break
            except ConnectionRefusedError:
                if time.time() - t0 > self.timeout_s:
                    raise
                time.sleep(0.05)

    def get_target_feat(self, batch):
        self._connect()
        msg = {
            "cmd": "infer",
            "input_ids": batch["input_ids"],
        }
        if "attention_mask" in batch:
            msg["attention_mask"] = batch["attention_mask"]

        self._conn.send(msg)
        resp = self._conn.recv()
        return (
            resp["target_feat"],
            resp["target_logits"],
            resp["teacher_gold_prob"],
        )

    def close(self):
        try:
            if self._conn is not None:
                self._conn.close()
        finally:
            self._conn = None

    def __del__(self):
        self.close()


def start_target_server(
    target_model_name: str,
    cache_dir: str,
    device: str,
    base_port: int = 29550,
    target_layers: list = [-1],
):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    port = base_port + local_rank
    address = ("127.0.0.1", port)
    authkey = b"pard_target_feat"

    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_target_feat_server,
        args=(target_model_name, cache_dir, address, device, authkey, target_layers),
        daemon=True,
    )
    proc.start()

    def _cleanup():
        try:
            conn = Client(address, authkey=authkey)
            conn.send({"cmd": "quit"})
            conn.close()
        except Exception:
            pass

        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass

    atexit.register(_cleanup)
    return proc, address, authkey


# =========================
# util
# =========================
def load_lm_head_full(path, cache_dir=None):
    model = AutoModelForCausalLM.from_pretrained(path, cache_dir=cache_dir)
    lm_head = model.get_output_embeddings()
    for p in lm_head.parameters():
        p.requires_grad_(False)
    return lm_head


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


PARD2_TRAINING_ONLY_CONFIG_KEYS = {
    "ce_alpha",
    "kd_alpha",
    "kd_temperature",
    "prev_prob_loss",
    "target_feat_mask",
}

PARD2_LEGACY_CONFIG_KEYS = {
    "feat_scale",
    "ml",
    "pard_scale",
    "pard_target_dim",
    "pard2_ml",
    "proj_bias",
    "target_layers",
}


def _get_config_attr(cfg, *names, default=None):
    for name in names:
        if hasattr(cfg, name):
            return getattr(cfg, name)
    return default


def _remove_config_attr(cfg, name: str):
    if hasattr(cfg, name):
        try:
            delattr(cfg, name)
        except AttributeError:
            pass
    if hasattr(cfg, "__dict__"):
        cfg.__dict__.pop(name, None)


def _load_target_layers_from_training_config(checkpoint_dir: str):
    candidates = [
        os.path.join(checkpoint_dir, "config.yaml"),
        os.path.join(os.path.dirname(checkpoint_dir), "config.yaml"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                train_cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning(f"Failed to read target_layers from {path}: {exc}")
            continue

        target_layers = train_cfg.get("general", {}).get("target_layers", None)
        if target_layers is not None:
            return [int(layer) for layer in target_layers]
    return None


def prepare_pard2_infer_config(
    cfg,
    checkpoint_dir: str = "",
    scale=None,
    proj_bias=None,
    target_dim=None,
    target_layers=None,
):
    if cfg is None:
        return None

    scale = scale if scale is not None else _get_config_attr(
        cfg, "pard2_scale", "feat_scale", "pard_scale", default=0.02
    )
    proj_bias = proj_bias if proj_bias is not None else _get_config_attr(
        cfg, "pard2_proj_bias", "proj_bias", default=False
    )
    target_dim = target_dim if target_dim is not None else _get_config_attr(
        cfg, "pard2_target_dim", "pard_target_dim", default=4096
    )
    target_layers = target_layers if target_layers is not None else _get_config_attr(
        cfg, "pard2_target_layers", "target_layers", default=None
    )
    if target_layers is None:
        use_multi_layer = _get_config_attr(
            cfg, "pard2_ml", "ml", default=bool("_ml" in checkpoint_dir or int(target_dim) > 10000)
        )
        target_layers = [-1, -8, -16, -24] if use_multi_layer else [-1]

    for key in PARD2_TRAINING_ONLY_CONFIG_KEYS | PARD2_LEGACY_CONFIG_KEYS:
        _remove_config_attr(cfg, key)

    cfg.pard2 = True
    cfg.spd_type = "pard2"
    cfg.pard2_scale = float(scale)
    cfg.pard2_proj_bias = bool(proj_bias)
    cfg.pard2_target_dim = int(target_dim)
    cfg.pard2_target_layers = [int(layer) for layer in target_layers]
    return cfg


def get_draft_model_config(model):
    model = unwrap_model(model)
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        return model.base_model.config
    if hasattr(model, "config"):
        return model.config
    return None


def save_draft_config(model, save_dir: str):
    cfg = get_draft_model_config(model)
    if cfg is None:
        logger.warning(f"Cannot find model.config, skip saving config.json to: {save_dir}")
        return
    os.makedirs(save_dir, exist_ok=True)
    cfg.save_pretrained(save_dir)


def _convert_checkpoint_for_infer_impl(checkpoint_dir: str):
    model_file = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"missing model.safetensors: {model_file}")

    model_path = os.path.join(checkpoint_dir, "pard_model")
    warp_model_path = os.path.join(checkpoint_dir, "pard_warp_model")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(warp_model_path, exist_ok=True)

    state = load_file(model_file, device="cpu")

    base_sd = {}
    warp_sd = {}
    prefix = "base_model."
    for key, value in state.items():
        if key.startswith(prefix):
            base_sd[key[len(prefix):]] = value
        elif key.startswith("target_proj."):
            warp_sd[key] = value

    base_params = sum(t.numel() for t in base_sd.values())
    warp_params = sum(t.numel() for t in warp_sd.values())

    metadata = {
        "source": model_file,
        "format": "pt",
    }
    base_out = os.path.join(model_path, "model.safetensors")
    warp_out = os.path.join(warp_model_path, "model.safetensors")
    save_file(base_sd, base_out, metadata=metadata)
    save_file(warp_sd, warp_out, metadata=metadata)
    torch.save(warp_sd, os.path.join(model_path, "warp_model.bin"))



    cfg = AutoConfig.from_pretrained(checkpoint_dir)
    prepare_pard2_infer_config(
        cfg,
        checkpoint_dir=checkpoint_dir,
        target_layers=_load_target_layers_from_training_config(checkpoint_dir),
    )
    cfg.save_pretrained(model_path)

    return {
        "model_path": model_path,
        "base_tensors": len(base_sd),
        "base_params": base_params,
        "warp_tensors": len(warp_sd),
        "warp_params": warp_params,
    }


def _convert_checkpoint_for_infer_worker(checkpoint_dir: str, result_queue):
    try:
        result_queue.put({"ok": True, "result": _convert_checkpoint_for_infer_impl(checkpoint_dir)})
    except BaseException as exc:
        import traceback

        result_queue.put(
            {
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )


def convert_checkpoint_for_infer(checkpoint_dir: str, tag: str):
    model_file = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(model_file):
        logger.warning(f"[auto_convert:{tag}] missing model.safetensors, skip: {model_file}")
        return

    logger.info(f"[auto_convert:{tag}] converting in a CPU subprocess: {checkpoint_dir}")
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_convert_checkpoint_for_infer_worker,
        args=(checkpoint_dir, result_queue),
        daemon=False,
    )
    proc.start()
    proc.join()

    try:
        payload = result_queue.get_nowait()
    except Empty:
        payload = None

    if proc.exitcode != 0:
        logger.error(f"[auto_convert:{tag}] subprocess failed with exitcode={proc.exitcode}")
        if payload and not payload.get("ok", False):
            logger.error(f"[auto_convert:{tag}] {payload['error']}\n{payload['traceback']}")
        return

    if not payload:
        logger.error(f"[auto_convert:{tag}] subprocess finished without returning a result")
        return

    if not payload.get("ok", False):
        logger.error(f"[auto_convert:{tag}] {payload['error']}\n{payload['traceback']}")
        return

    result = payload["result"]
    logger.info(
        f"[auto_convert:{tag}] split weights: "
        f"base_tensors={result['base_tensors']}, base_params={result['base_params']}, "
        f"warp_tensors={result['warp_tensors']}, warp_params={result['warp_params']}"
    )
    logger.info(f"[auto_convert:{tag}] infer draft path: {result['model_path']}")


def resolve_resume_checkpoint(output_dir: str, general_cfg: dict):
    resume_from_checkpoint = general_cfg.get("resume_from_checkpoint", None)
    auto_resume = bool(general_cfg.get("auto_resume", True))

    if resume_from_checkpoint:
        ckpt_path = resume_from_checkpoint

        if isinstance(ckpt_path, str) and ckpt_path.lower() == "latest":
            ckpt_path = get_last_checkpoint(output_dir)
        elif not os.path.isabs(ckpt_path):
            local_candidate = os.path.join(output_dir, ckpt_path)
            if os.path.isdir(local_candidate):
                ckpt_path = local_candidate

        if ckpt_path and os.path.isdir(ckpt_path):
            logger.info(f"Resume from checkpoint: {ckpt_path}")
            return ckpt_path

        logger.warning(
            f"general.resume_from_checkpoint={resume_from_checkpoint} not found, "
            f"will try auto_resume={auto_resume}."
        )

    if auto_resume:
        last_ckpt = get_last_checkpoint(output_dir)
        if last_ckpt is not None:
            logger.info(f"Auto resume from latest checkpoint: {last_ckpt}")
            return last_ckpt

    logger.info("No valid checkpoint found, train from scratch.")
    return None


class DraftConfigSaveCallback(TrainerCallback):
    def __init__(self, output_dir: str, draft_config_subdir: str = "draft_model_config"):
        self.output_dir = output_dir
        self.draft_config_dir = os.path.join(output_dir, draft_config_subdir)

    def _save(self, model, save_dir):
        if model is None:
            return
        save_draft_config(model, save_dir)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        self._save(model, args.output_dir)
        self._save(model, self.draft_config_dir)

    def on_save(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self._save(model, checkpoint_dir)
        self._save(model, self.draft_config_dir)
        convert_checkpoint_for_infer(checkpoint_dir, "checkpoint")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        self._save(model, args.output_dir)
        self._save(model, self.draft_config_dir)


# =========================
# model wrapper
# =========================
class TargetFeatEmbedWarp(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        proj_bias: bool = False,
        scale: float = 1.0,
        target_dim: int = 0,
        target_head: nn.Module = None,
        ce_alpha: float = 0.3,
        kd_alpha: float = 1.0,
        kd_temperature: float = 1.0,
        target_feat_mask: float = 0.5,
        prev_prob_loss: bool = True,
    ):
        super().__init__()

        self.base_model = base_model
        self.target_head = target_head
        self.scale = float(scale)
        self.target_dim = int(target_dim)

        self.ce_alpha = float(ce_alpha)
        self.kd_alpha = float(kd_alpha)
        self.kd_temperature = float(kd_temperature)
        self.target_feat_mask = target_feat_mask
        self.prev_prob_loss = prev_prob_loss

        if self.target_head is not None:
            for p in self.target_head.parameters():
                p.requires_grad_(False)

        self.target_proj = nn.Linear(
            self.target_dim,
            self.base_model.config.hidden_size,
            bias=proj_bias,
        )

        self.config = getattr(base_model, "config", None)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            return self.base_model.gradient_checkpointing_enable(*args, **kwargs)
        return None

    def gradient_checkpointing_disable(self, *args, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            return self.base_model.gradient_checkpointing_disable(*args, **kwargs)
        return None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        inputs_embeds=None,
        target_feat=None,
        target_logits=None,  
        prev_prob=None,
        **kwargs,
    ):
        # 1) word embedding
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Need either input_ids or inputs_embeds")
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
    
        # 2) fuse teacher feature
        if target_feat is not None:
            tf = target_feat.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            tf_proj = self.target_proj(tf)
            tf_proj = tf_proj * self.scale
    
            if tf_proj.shape[:2] != inputs_embeds.shape[:2]:
                raise ValueError(
                    f"target_feat seq mismatch: proj={tuple(tf_proj.shape)} vs embeds={tuple(inputs_embeds.shape)}"
                )
            bsz = tf_proj.shape[0]
            keep_mask = (torch.rand(bsz, 1, 1, device=tf_proj.device) > self.target_feat_mask).to(tf_proj.dtype)
            tf_proj = tf_proj * keep_mask
            inputs_embeds = inputs_embeds + tf_proj
    
        kwargs.pop("num_items_in_batch", None)
    
        # 3) forward student model
        outputs = self.base_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=None, 
            **kwargs,
        )
    
        student_logits = outputs.logits  # [B, T, V]
        ce_loss = None
        kd_loss = None
        token_weight = None
    
        # 4) weighted CE
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()  # [B, T-1]
            valid_mask = (shift_labels != -100).to(student_logits.dtype)
    
            if prev_prob is not None and self.prev_prob_loss:
                shift_prev_prob = prev_prob[..., 1:].to(
                    device=student_logits.device,
                    dtype=student_logits.dtype,
                )
                token_weight = valid_mask * shift_prev_prob
            else:
                token_weight = valid_mask
    
            shift_student_logits = student_logits[..., :-1, :].contiguous().float()
            vocab_size = shift_student_logits.size(-1)
    
            ce_per_token = F.cross_entropy(
                shift_student_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view_as(shift_labels)
    
            denom = token_weight.sum().clamp_min(1e-6)
            ce_loss = (ce_per_token * token_weight).sum() / denom
    
        # 5) weighted KD
        if target_logits is not None and self.target_head is not None:
            teacher_hidden = target_logits.to(
                device=student_logits.device,
                dtype=student_logits.dtype,
            )
            teacher_logits = self.target_head(teacher_hidden)
    
            T = float(self.kd_temperature)
    
            shift_teacher_logits = teacher_logits[..., :-1, :].contiguous().float()
            shift_student_logits = student_logits[..., :-1, :].contiguous().float()
    
            teacher_prob = F.softmax(shift_teacher_logits / T, dim=-1)
            student_log_prob = F.log_softmax(shift_student_logits / T, dim=-1)
    
            token_kd = F.kl_div(
                student_log_prob,
                teacher_prob,
                reduction="none",
            ).sum(dim=-1) * (T ** 2)
    
            if labels is not None:
                kd_weight = token_weight.to(token_kd.dtype)
                kd_loss = (token_kd * kd_weight).sum() / kd_weight.sum().clamp_min(1e-6)
            elif prev_prob is not None and self.prev_prob_loss:
                kd_weight = prev_prob[..., 1:].to(
                    device=token_kd.device,
                    dtype=token_kd.dtype,
                )
                kd_loss = (token_kd * kd_weight).sum() / kd_weight.sum().clamp_min(1e-6)
            else:
                kd_loss = token_kd.mean()
    
        final_loss = None
        if ce_loss is not None and kd_loss is not None:
            final_loss = ce_loss * self.ce_alpha + kd_loss * self.kd_alpha
        elif ce_loss is not None:
            final_loss = ce_loss
        elif kd_loss is not None:
            final_loss = kd_loss
    
        if final_loss is None:
            raise ValueError(
                "No loss was computed. Check labels / target_logits / prev_prob inputs."
            )

        try:
            outputs["loss"] = final_loss
        except Exception:
            pass
    
        try:
            outputs.loss = final_loss
        except Exception:
            pass
    
        return outputs
# =========================
# collator
# =========================
class WarpDataCollator:
    def __init__(
        self,
        tokenizer,
        target_address=None,
        target_authkey=None,
        use_target_model=False,
        para_num=4,
        unused_tokenids=None,
        down_sample_ration=1.0,
        down_sample_ration_min=0.0,
        end_token_id=128007,
    ):
        self.tokenizer = tokenizer
        self.target_address = target_address
        self.target_authkey = target_authkey
        self.use_target_model = bool(use_target_model)

        self.para_num = int(para_num)
        self.unused_tokenids = unused_tokenids or []
        self.down_sample_ration = float(down_sample_ration)
        self.down_sample_ration_min = float(down_sample_ration_min)
        self.end_token_id = end_token_id

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        self._client = None

        if self.para_num > 1 and len(self.unused_tokenids) < self.para_num - 1:
            raise ValueError(
                f"unused_tokenids length ({len(self.unused_tokenids)}) < para_num-1 ({self.para_num - 1})"
            )

    def _get_client(self):
        if self._client is None:
            self._client = TargetFeatClient(self.target_address, self.target_authkey)
        return self._client

    def _build_prev_prob(self, teacher_gold_prob: torch.Tensor, para_num: int):
        teacher_gold_prob = teacher_gold_prob.float()
        ones = torch.ones_like(teacher_gold_prob, dtype=torch.float32)
        blocks = []

        for i in range(para_num):
            block = ones.clone()
            for s in range(1, i + 1):
                shifted = torch.cat(
                    [ones[:, :s], teacher_gold_prob[:, :-s]],
                    dim=1,
                )
                block = block * shifted
            blocks.append(block)

        return torch.cat(blocks, dim=1)

    def __call__(self, features):
        data = self.data_collator(features)

        target_feat = None
        target_logits = None
        teacher_gold_prob = None

        if self.use_target_model:
            with torch.no_grad():
                target_feat, target_logits, teacher_gold_prob = self._get_client().get_target_feat(data)

        labels = data["labels"]
        IGNORE = -100

        # 只保留最后一个 end_token_id 之后的 loss
        if self.end_token_id is not None:
            B, _ = labels.shape
            for i in range(B):
                pos = (labels[i] == self.end_token_id).nonzero(as_tuple=False).flatten()
                if pos.numel() == 0:
                    continue
                last_end = pos[-1].item()
                labels[i, :last_end] = IGNORE

        data["labels"] = labels

        para_num = self.para_num
        single_length = data["input_ids"].shape[1]


        if teacher_gold_prob is None:
            teacher_gold_prob = torch.ones_like(data["labels"], dtype=torch.float32)

        if para_num == 1:
            new_data = {
                "input_ids": data["input_ids"],
                "attention_mask": data.get("attention_mask", None),
                "labels": data["labels"],
                "prev_prob": torch.ones_like(data["labels"], dtype=torch.float32),
            }
            if self.use_target_model:
                new_data["target_feat"] = target_feat
                new_data["target_logits"] = target_logits
            return new_data

        tgt_len = single_length * para_num
        mask = torch.full((tgt_len, tgt_len), torch.finfo(torch.float32).min)
        mask_cond = torch.arange(mask.size(-1))

        tmp_mask = (mask_cond == (mask_cond).view(mask.size(-1), 1))
        for i in range(para_num):
            tmp_mask = tmp_mask | (
                mask_cond == (mask_cond - single_length * i - i).view(mask.size(-1), 1)
            )
            tmp_mask = tmp_mask | (
                (mask_cond < (mask_cond - i * single_length - (i - 1)).view(mask.size(-1), 1))
                & (mask_cond < (i + 1) * single_length).view(-1, 1)
            )

        mask.masked_fill_(tmp_mask, 0)
        mask = mask[None, None, :, :]  # [1,1,T,T]

        bs = data["input_ids"].shape[0]

        new_labels = torch.cat(
            [
                torch.cat(
                    [data["labels"][:, :i] * 0 - 100, data["labels"][:, i:]],
                    dim=1,
                )
                for i in range(para_num)
            ],
            dim=1,
        )

        prev_prob = self._build_prev_prob(teacher_gold_prob, para_num)
        prev_prob = prev_prob.masked_fill(new_labels == IGNORE, 0.0)

        new_data = {
            "input_ids": torch.cat(
                [
                    data["input_ids"],
                    torch.cat(
                        [
                            data["input_ids"] * 0 + self.unused_tokenids[i]
                            for i in range(para_num - 1)
                        ],
                        dim=1,
                    ),
                ],
                dim=1,
            ),
            "attention_mask": torch.cat([mask for _ in range(bs)], dim=0),
            "position_ids": torch.cat(
                [torch.arange(single_length, dtype=torch.long) for _ in range(para_num)],
                dim=0,
            ).unsqueeze(0).repeat(bs, 1),
            "labels": new_labels,
            "prev_prob": prev_prob,
        }

        if self.use_target_model:
            new_data["target_logits"] = torch.cat(
                [target_logits for _ in range(para_num)],
                dim=1,
            )
            new_data["target_feat"] = torch.cat(
                [
                    torch.cat(
                        [
                            target_feat[:, :i] * 0,
                            target_feat[:, : target_feat.shape[1] - i],
                        ],
                        dim=1,
                    )
                    for i in range(para_num)
                ],
                dim=1,
            )

        # COD down sample
        if self.down_sample_ration != 1 and para_num > 1:
            index_mask = torch.zeros(para_num, single_length, dtype=torch.bool)
            index_mask[0, :] = True
            prev_indices = torch.arange(single_length)

            for i in range(1, para_num):
                num_ones = int(
                    single_length
                    * max(self.down_sample_ration ** i, self.down_sample_ration_min)
                )
                if num_ones <= 0:
                    break

                num_ones = min(num_ones, len(prev_indices))
                selected_indices = prev_indices[torch.randperm(len(prev_indices))[:num_ones]]
                index_mask[i, selected_indices] = True
                prev_indices = (selected_indices + 1) % single_length

            index_mask = index_mask.reshape(-1)
            indices = index_mask.nonzero(as_tuple=True)[0]

            filtered_data = {
                "input_ids": new_data["input_ids"][:, indices].contiguous(),
                "position_ids": new_data["position_ids"][:, indices].contiguous(),
                "labels": torch.roll(
                    torch.roll(new_data["labels"], shifts=-1, dims=1)[:, indices],
                    shifts=1,
                    dims=1,
                ).contiguous(),
                "prev_prob": torch.roll(
                    torch.roll(new_data["prev_prob"], shifts=-1, dims=1)[:, indices],
                    shifts=1,
                    dims=1,
                ).contiguous(),
                "attention_mask": new_data["attention_mask"][:, :, indices, :][:, :, :, indices].contiguous(),
            }

            if self.use_target_model:
                filtered_data["target_feat"] = new_data["target_feat"][:, indices].contiguous()
                filtered_data["target_logits"] = new_data["target_logits"][:, indices].contiguous()

            new_data = filtered_data

        return new_data


# =========================
# main
# =========================
@click.command()
@click.option("-c", "--config_path", required=True, type=str)
def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config_general = config.get("general", {})
    config_train = config.get("train", {})
    config_data = config.get("data", {})
    config_loss = config.get("loss", {})
    config_collator = config.get("collator", {})

    model_name = config_general["model_name"]
    cache_dir = config_general.get("cache_dir", "cache")
    target_model_name = config_general.get("target_model_name", None)

    exp_name = config_general.get("exp_name", None)
    exp_name = model_name.split("/")[-1] + "_" + (
        exp_name if exp_name is not None else os.path.basename(os.getcwd())
    )

    output_dir = os.path.join(config_general["save_base_path"], exp_name)
    os.makedirs(output_dir, exist_ok=True)

    shutil.copy2(config_path, os.path.join(output_dir, "config.yaml"))

    logger.info(f"[model_path]: {model_name}")
    logger.info(f"[save_path]: {output_dir}")

    # ---------- build model ----------
    base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

    target_model_enabled = target_model_name is not None
    target_address = None
    target_authkey = None
    server_proc = None

    if target_model_enabled:
        target_cfg = AutoConfig.from_pretrained(target_model_name, cache_dir=cache_dir)
        feat_scale = float(config_general.get("feat_scale", 0.01))
        proj_bias = bool(config_general.get("proj_bias", False))
        target_layers = config_general.get("target_layers", [-1])
        target_dim = target_cfg.hidden_size * len(target_layers)

        model = TargetFeatEmbedWarp(
            base_model=base_model,
            proj_bias=proj_bias,
            scale=feat_scale,
            target_dim=target_dim,
            target_head=load_lm_head_full(target_model_name, cache_dir=cache_dir),
            ce_alpha=float(config_loss.get("ce_alpha", 0.3)),
            kd_alpha=float(config_loss.get("kd_alpha", 1.0)),
            kd_temperature=float(config_loss.get("kd_temperature", 1.0)),
            target_feat_mask=float(config_general.get("target_feat_mask", 0.0)),
            prev_prob_loss=bool(config_general.get("prev_prob_loss", True)),
        )

        prepare_pard2_infer_config(
            model.config,
            checkpoint_dir=output_dir,
            scale=feat_scale,
            proj_bias=proj_bias,
            target_dim=target_dim,
            target_layers=target_layers,
        )

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        logger.info(f"[device]: {device}")

        logger.info(f'[target_layers]: {target_layers}')
        server_proc, target_address, target_authkey = start_target_server(
            target_model_name=target_model_name,
            cache_dir=cache_dir,
            device=device,
            base_port=int(config_general.get("target_server_base_port", 29550)),
            target_layers=target_layers,
        )
    else:
        model = base_model

    # ---------- tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    unused_tokenids = config_general.get("unused_tokenids", [])
    model.config.pard_token = unused_tokenids[0] if len(unused_tokenids) > 0 else -1
    model.config.spd_type = "pard2"
    if target_model_enabled:
        prepare_pard2_infer_config(
            model.config,
            checkpoint_dir=output_dir,
            target_layers=target_layers,
        )

    draft_config_subdir = config_general.get("draft_config_subdir", "draft_model_config")
    draft_config_dir = os.path.join(output_dir, draft_config_subdir)
    save_draft_config(model, output_dir)
    save_draft_config(model, draft_config_dir)

    # ---------- dataset ----------
    dataset = load_dataset(
        "json",
        split="train",
        data_files=config_general["data_path"],
    )

    def formatting_prompts_func(example):
        prompt_template = config_data["prompt_template"][config_data["prompt_type"]]

        if "conversation" in example:
            conversations = example["conversation"]

            # batched
            if isinstance(conversations, list) and len(conversations) > 0 and isinstance(conversations[0], list):
                output_texts = []
                for conv in conversations:
                    prompt = conv[0]["human"]
                    answer = conv[0]["assistant"]
                    text = prompt_template.format(prompt=prompt, answer=answer)
                    output_texts.append(text)
                return output_texts

            # single sample
            prompt = conversations[0]["human"]
            answer = conversations[0]["assistant"]
            return prompt_template.format(prompt=prompt, answer=answer)

        if "text" in example:
            texts = example["text"]
            if isinstance(texts, list):
                return texts
            return texts

        raise ValueError("Dataset example must contain either 'conversation' or 'text'.")

    # ---------- trainer config ----------
    train_kwargs = dict(config_train)
    train_kwargs.setdefault("bf16", True)
    sft_config = SFTConfig(
        output_dir=output_dir,
        **train_kwargs,
    )

    if getattr(sft_config, "gradient_checkpointing", False):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    # ---------- collator ----------
    data_collator = WarpDataCollator(
        tokenizer=tokenizer,
        target_address=target_address,
        target_authkey=target_authkey,
        use_target_model=target_model_enabled,
        para_num=int(config_general.get("para_num", 1)),
        unused_tokenids=unused_tokenids,
        down_sample_ration=float(config_general.get("down_sample_ration", 1.0)),
        down_sample_ration_min=float(config_general.get("down_sample_ration_min", 0.0)),
        end_token_id=config_collator.get("end_token_id", 128007),
    )

    # ---------- trainer ----------
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=data_collator,
        args=sft_config,
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
        callbacks=[DraftConfigSaveCallback(output_dir, draft_config_subdir=draft_config_subdir)],
    )

    # ---------- resume ----------
    resume_ckpt = resolve_resume_checkpoint(output_dir, config_general)

    # ---------- train ----------
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ---------- save final ----------
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_draft_config(model, output_dir)
    save_draft_config(model, draft_config_dir)
    convert_checkpoint_for_infer(output_dir, "final")

    logger.info(f"[final_save_path]: {output_dir}")


if __name__ == "__main__":
    main()
