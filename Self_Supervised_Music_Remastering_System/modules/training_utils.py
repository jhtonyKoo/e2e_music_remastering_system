"""
    Utility file for trainers
"""
import os
import shutil
from glob import glob

import torch
import torch.distributed as dist



''' checkpoint functions '''
# saves checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir, name, model_name):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_state = {    
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch
        }
    checkpoint_path = os.path.join(checkpoint_dir,'{}_{}_{}.pt'.format(name, model_name, epoch))
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))


# reload model weights from checkpoint file
def reload_ckpt(args, network, optimizer, scheduler, gpu, model_name, manual_reload_name=None, manual_reload=False, epoch=None, fit_sefa=False):
    if manual_reload:
        reload_name = manual_reload_name
    else:
        reload_name = args.name
    ckpt_dir = args.output_dir + reload_name + "/ckpt/"
    temp_ckpt_dir = f'{args.output_dir}{reload_name}/ckpt_temp/'
    reload_epoch = epoch
    # find best or latest epoch
    if epoch==None:
        reload_epoch_temp = 0
        reload_epoch_ckpt = 0
        if len(os.listdir(temp_ckpt_dir))!=0:
            reload_epoch_temp = find_best_epoch(temp_ckpt_dir)
        if len(os.listdir(ckpt_dir))!=0:
            reload_epoch_ckpt = find_best_epoch(ckpt_dir)
        if reload_epoch_ckpt >= reload_epoch_temp:
            reload_epoch = reload_epoch_ckpt
        else:
            reload_epoch = reload_epoch_temp
            ckpt_dir = temp_ckpt_dir
    else:
        if os.path.isfile(f"{temp_ckpt_dir}{reload_epoch}/{reload_name}_{model_name}_{reload_epoch}.pt"):
            ckpt_dir = temp_ckpt_dir
    # reloading weight
    if model_name==None:
        resuming_path = f"{ckpt_dir}{reload_epoch}/{reload_name}_{reload_epoch}.pt"
    else:
        resuming_path = f"{ckpt_dir}{reload_epoch}/{reload_name}_{model_name}_{reload_epoch}.pt"
    if gpu==0:
        print("===Resume checkpoint from: {}===".format(resuming_path))
    loc = 'cuda:{}'.format(gpu)
    checkpoint = torch.load(resuming_path, map_location=loc)
    start_epoch = 0 if manual_reload and not fit_sefa else checkpoint["epoch"]
    network.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if gpu==0:
        print("=> loaded checkpoint '{}' (epoch {})".format(resuming_path, checkpoint['epoch']))
    return start_epoch


# find best epoch for reloading current model
def find_best_epoch(input_dir):
    cur_epochs = glob("{}*".format(input_dir))
    return find_by_name(cur_epochs)


# sort string epoch names by integers
def find_by_name(epochs):
    int_epochs = []
    for e in epochs:
        int_epochs.append(int(os.path.basename(e)))
    int_epochs.sort()
    return (int_epochs[-1])


# remove ckpt files
def remove_ckpt(cur_ckpt_path_dir, leave=2):
    ckpt_nums = [int(i) for i in os.listdir(cur_ckpt_path_dir)]
    ckpt_nums.sort()
    del_num = len(ckpt_nums) - leave
    cur_del_num = 0
    while del_num > 0:
        shutil.rmtree("{}{}".format(cur_ckpt_path_dir, ckpt_nums[cur_del_num]))
        del_num -= 1
        cur_del_num += 1



'''
    Gather layer for multi-GPU training procedure
    below source code is a replication from the github repository - https://github.com/Spijkervet/SimCLR
    the original implementation can be found here: https://github.com/Spijkervet/SimCLR/blob/cd85c4366d2e6ac1b0a16798b76ac0a2c8a94e58/simclr/modules/gather.py#L5
'''
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
            for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
