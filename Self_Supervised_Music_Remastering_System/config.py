import argparse
import yaml

parser = argparse.ArgumentParser()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    params = parser.parse_args()
    return params


def print_args(params):
    info = '\n[args]\n'
    for sub_args in parser._action_groups:
        if sub_args.title in ['positional arguments', 'optional arguments']:
            continue
        size_sub = len(sub_args._group_actions)
        info += f'  {sub_args.title} ({size_sub})\n'
        for i, arg in enumerate(sub_args._group_actions):
            prefix = '-'
            info += f'      {prefix} {arg.dest:20s}: {getattr(params, arg.dest)}\n'
    info += '\n'
    print(info)





reload_args = parser.add_argument_group('Reload args')
reload_args.add_argument('--ckpt_dir', type=str, default="model_checkpoints/", help="reloading network architecture")
# reload_args.add_argument('--ckpt_path', type=str, default="mastering_cloner.pt", help="reloading network architecture")
reload_args.add_argument('--ckpt_path', type=str, default="wider3_newmee_fixed_af_mssddsp_re_cloner_1000.pt", help="reloading network architecture")
reload_args.add_argument('--ckpt_path_feat', type=str, default="music_effects_encoder_1000.pt", help="reloading network architecture")


base_args = parser.add_argument_group('Base args')
base_args.add_argument('--output_dir', type=str, default="inference_samples/")
base_args.add_argument('--data_dir_test', type=str, default="inference_samples/")
base_args.add_argument('--data_dir_feat_test', type=str)
# base_args.add_argument('--data_dir', type=str)
# base_args.add_argument('--data_dir_feat', type=str)
base_args.add_argument('--data_dir', type=str, default="/data3/btsd/")
base_args.add_argument('--data_dir_feat', type=str, default="/data2/jamendo/wav_44/")

train_args = parser.add_argument_group('Train args')
train_args.add_argument('--random_seed', type=int, default=111)
train_args.add_argument('--num_epochs', type=int, default=1001)
train_args.add_argument('--batch_size', type=int, default=8)
train_args.add_argument('--using_loss_contrastive', type=str, default=["ntxent"], help="usable loss functions are: 1. ntxent")
train_args.add_argument('--using_loss_wave', type=str, default=["gain", "multi_scale_spectral_midside", "multi_scale_spectral_ori"], help="usable loss functions are: 1. l1   2. mse   3. multi_scale_spectral   4. midside   5. multi_scale_spectral_midside   6. gain")
train_args.add_argument('--using_loss_gan', type=str, default=["hinge"], help="usable loss functions are: 1. hinge")

data_args = parser.add_argument_group('Data args')
data_args.add_argument('--sample_rate', type=int, default=44100)
data_args.add_argument('--segment_length', type=int, default=2**17)
data_args.add_argument('--max_segment_length_feat', type=int, default=44100*5)
data_args.add_argument('--reference_length', type=int, default=None)

network_args = parser.add_argument_group('Network args')
network_args.add_argument('--network_arc_feat', type=str, default='Effects_Encoder')
network_args.add_argument('--network_options_feat', type=str, default='default', help="network options for each feature extractors: see configs.yaml for detailed settings")
network_args.add_argument('--network_arc', type=str, default='Mastering_Cloner')
network_args.add_argument('--network_options', type=str, default='default', help="network options for each architecture: see configs.yaml for detailed settings")
network_args.add_argument('--network_arc_disc', type=str, default='Projection_Discriminator_2D')
network_args.add_argument('--network_options_disc', type=str, default='default', help="network options for discriminator: see configs.yaml for detailed settings")

hyperparam_args = parser.add_argument_group('Hyperparameter args')
hyperparam_args.add_argument('--optimizer', type=str, default='Adam', help="using optimizer: 1. Adam   2. RAdam")
hyperparam_args.add_argument('--learning_rate', type=float, default=2e-4)
hyperparam_args.add_argument('--learning_rate_feat', type=float, default=2e-4)
hyperparam_args.add_argument('--weight_decay', default=1e-3, type=float, help="learning rate decay factor")
hyperparam_args.add_argument('--patience', default=2, type=int, help='patience')
hyperparam_args.add_argument('--eps', default=1e-7, type=float, help="epsilon value for preventing 'nan' values")
hyperparam_args.add_argument('--temperature', default=0.5, type=float, help="temperature value for NT-Xent loss")

gpu_args = parser.add_argument_group('GPU args')
gpu_args.add_argument('--n_nodes', default=1, type=int)
gpu_args.add_argument('--workers', default=4, type=int)
gpu_args.add_argument('--rank', default=0, type=int, help='ranking within the nodes')

args = get_args()

# load network configurations
with open('Self_Supervised_Music_Remastering_System/configs.yaml', 'r') as f:
    configs = yaml.load(f)
args.cfg = configs[args.network_arc][args.network_options]
args.cfg_feat = configs[args.network_arc_feat][args.network_options_feat]
args.cfg_disc = configs[args.network_arc_disc][args.network_options_disc]


if __name__ == '__main__':
    args = get_args()
    print_args(args)