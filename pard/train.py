import click
import os

@click.command()
@click.option('-c', '--config_path')
@click.option('-d', '--device', default='0,1,2,3,4,5,6,7')
@click.option('-p', '--port', default='29501')
def main(config_path, device, port):
    ds_yaml = 'config/train/deepspeed_zero1.yaml'
    cmd = f'''CUDA_VISIBLE_DEVICES='{device}' accelerate launch --main_process_port {port} --config_file {ds_yaml} --num_processes {len(device.split(','))} pard/pard_train.py -c {config_path}'''
    print(f'\n\ncmd: \n{cmd}\n\n')
    os.system(cmd)

if __name__ == '__main__':
    main()
