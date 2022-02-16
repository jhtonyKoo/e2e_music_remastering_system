""" 
    Source code of inference procedure of the task 'End-to-end Music Remastering System Using Self-supervised and Adversarial Training'
"""
import os
import soundfile as sf

import torch
from torch.utils.data import DataLoader

from Self_Supervised_Music_Remastering_System.data_loader import *
from Self_Supervised_Music_Remastering_System.networks import *




class Remastering_System_Inference:
    def __init__(self, args):
        self.device = torch.device("cpu")
        # if torch.cuda.is_available():
        #     self.device = torch.device("cuda:0")
        
        # inference computational hyperparameters
        self.segment_length = args.segment_length
        self.segment_length_feat = args.max_segment_length_feat
        self.batch_size = args.batch_size
        self.sample_rate = args.sample_rate

        # directory configuration
        self.output_dir = args.output_dir

        # data loader
        inference_dataset = Song_Dataset_Inference(args=args)
        collate_class = Collate_Variable_Length_Segments(args)
        self.data_loader = DataLoader(inference_dataset, \
                                        batch_size=1, \
                                        shuffle=False, \
                                        num_workers=args.workers, \
                                        drop_last=False)


        # load model and its checkpoint weights
        self.mastering_cloner = Mastering_Cloner_Wave_Unet(args.cfg).to(self.device)
        self.music_effects_encoder = Music_Effects_Encoder(args.cfg_feat).to(self.device)

        # reload saved model weights
        assert os.path.exists(args.ckpt_dir+args.ckpt_path), \
                f"make sure checkpoint file for the Mastering Cloner named '{args.ckpt_path}' is under directory '{args.ckpt_dir}'"
        assert os.path.exists(args.ckpt_dir+args.ckpt_path_feat), \
                f"make sure checkpoint file for the Mastering Cloner named '{args.ckpt_path_feat}' is under directory '{args.ckpt_dir}'"
        self.reload_weights(self.mastering_cloner, \
                                args.ckpt_dir+args.ckpt_path, \
                                self.device)
        print("---reloaded checkpoint weights - Mastering Cloner---")
        self.reload_weights(self.music_effects_encoder, \
                                args.ckpt_dir+args.ckpt_path_feat, \
                                self.device)
        print("---reloaded checkpoint weights - Music Effects Encoder---")


    # reload model weights from the target checkpoint path
    def reload_weights(self, model, ckpt_path, device):
        checkpoint = torch.load(ckpt_path, map_location=device)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        # since our networks were trained using DistributedDataParallel,
        #   we need to remove the name `module.` in order to load weights when not using DDP
        for k, v in checkpoint["model"].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)


    # Infer full lengthed songs in the target directory
    def inference(self, ):
        for step, (whole_song_ori, whole_song_ref, song_name) in enumerate(self.data_loader):
            print(f"-------inference file name : {song_name[0]}-------")
            ''' segmentize whole songs into batch '''
            whole_batch_data_ori = self.batchwise_segmentization(whole_song_ori[0], song_name, segment_length=self.segment_length, discard_last=False)
            whole_batch_data_ref = self.batchwise_segmentization(whole_song_ref[0], song_name, segment_length=self.segment_length_feat, discard_last=True)

            ''' inference '''
            # first extract reference style embedding
            infered_ref_data_list = []
            for cur_ref_data in whole_batch_data_ref:
                cur_ref_data = cur_ref_data.to(self.device)
                # Music Effects Encoder inference
                with torch.no_grad():
                    self.music_effects_encoder.eval()
                    _, reference_feature = self.music_effects_encoder(cur_ref_data)
                infered_ref_data_list.append(reference_feature)
            # compute average value from the extracted exbeddings
            infered_ref_data = torch.stack(infered_ref_data_list)
            infered_ref_data_avg = torch.mean(infered_ref_data.reshape(infered_ref_data.shape[0]*infered_ref_data.shape[1], infered_ref_data.shape[2]), axis=0)

            # infer whole song
            infered_data_list = []
            for cur_data in whole_batch_data_ori:
                cur_data = cur_data.to(self.device)
                # Mastering Cloner inference
                with torch.no_grad():
                    self.mastering_cloner.eval()
                    infered_data = self.mastering_cloner(cur_data, infered_ref_data_avg.unsqueeze(0))
                infered_data_list.append(infered_data.cpu().detach())

            # combine back to whole song
            for cur_idx, cur_batch_infered_data in enumerate(infered_data_list):
                cur_infered_data_sequential = torch.cat(torch.unbind(cur_batch_infered_data, dim=0), dim=-1)
                fin_data_out = cur_infered_data_sequential if cur_idx==0 else torch.cat((fin_data_out, cur_infered_data_sequential), dim=-1)
            # final output
            fin_data_out = fin_data_out[:, :whole_song_ori.shape[-1]].numpy()

            # write output
            cur_out_dir = f"{self.output_dir}/{song_name[0]}/"
            os.makedirs(cur_out_dir, exist_ok=True)
            sf.write(f"{cur_out_dir}inferred_output.wav", fin_data_out.transpose(-1, -2), self.sample_rate, 'PCM_16')



    # function that segmentize an entire song into batch
    def batchwise_segmentization(self, target_song, song_name, segment_length, discard_last=False):
        assert target_song.shape[-1] >= self.segment_length, \
                f"Error : Insufficient duration!\n\t \
                Target song's length is shorter than segment length.\n\t \
                Song name : {song_name}\n\t \
                Consider changing the 'segment_length' or song with sufficient duration"

        # discard restovers (last segment)
        if discard_last:
            target_length = target_song.shape[-1] - target_song.shape[-1] % segment_length
            target_song = target_song[:, :target_length]
        # pad last segment
        else:
            pad_length = segment_length - target_song.shape[-1] % segment_length
            target_song = torch.cat((target_song, torch.zeros(2, pad_length)), axis=-1)
            
        # segmentize according to the given segment_length
        whole_batch_data = []
        batch_wise_data = []
        for cur_segment_idx in range(target_song.shape[-1]//segment_length):
            batch_wise_data.append(target_song[..., cur_segment_idx*segment_length:(cur_segment_idx+1)*segment_length])
            if len(batch_wise_data)==self.batch_size:
                whole_batch_data.append(torch.stack(batch_wise_data, dim=0))
                batch_wise_data = []
        if batch_wise_data:
            whole_batch_data.append(torch.stack(batch_wise_data, dim=0))

        return whole_batch_data





if __name__ == '__main__':
    """ Inference process for the task 'End-to-end Remastering System' """
    from Self_Supervised_Music_Remastering_System.config import args, print_args
    inf = Remastering_System_Inference(args)
    inf.inference()
