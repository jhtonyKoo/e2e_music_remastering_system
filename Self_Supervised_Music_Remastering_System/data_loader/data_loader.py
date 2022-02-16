"""
    Data Loaders of the task 'End-to-end Remastering System'
    includes training and inferencing models of the 'Mastering Cloner' and 'Music Effects Encoder'
"""
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import soundfile as sf
import wave
import time
import random

from glob import glob
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from mastering_manipulator.manipulator import Mastering_Effects_Manipulator
from loader_utils import *



'''
    Collate Functions
'''
class Collate_Variable_Length_Segments:
    def __init__(self, args):
        self.segment_length = args.segment_length
        self.max_segment_length_feat = args.max_segment_length_feat
        self.random_length = args.reference_length


    # collate function of manipulating duration of only one of the input segments
    def manipulate_duration_single_segment(self, batch):
        # inputs : segment_a1, segment_a2, segment_b2
        # only change duration of segment_b2

        # randomize the duration of current segment_b2
        max_length = self.max_segment_length_feat
        min_length = max_length // 2
        if self.random_length:
            random_length = self.random_length
        else:
            random_length = torch.randint(low=min_length, high=max_length, size=(1,))[0]

        segment_a1 = []
        segment_a2 = []
        segment_b2 = []
        segment_a = []
        for cur_item in batch:
            # set starting point
            if random_length==self.max_segment_length_feat:
                random_start_point = 0
            else:
                random_start_point = torch.randint(low=0, high=max_length-random_length, size=(1,))[0]
            # segmentize input
            segmentized_b = cur_item[2][:, random_start_point : random_start_point+random_length]
            segment_a1.append(cur_item[0])
            segment_a2.append(cur_item[1])
            segment_b2.append(segmentized_b)
            segment_a.append(cur_item[3])

        # returns segmentized inputs
        return torch.stack(segment_a1, dim=0), torch.stack(segment_a2, dim=0), torch.stack(segment_b2, dim=0), torch.stack(segment_a, dim=0)


    # collate function of manipulating duration of two different segments
    def manipulate_duration_both_segments(self, batch):
        # randomize current input length
        max_length = batch[0][0].shape[-1]
        min_length = max_length//2
        input_length_a, input_length_b = torch.randint(low=min_length, high=max_length, size=(2,))

        segment_a = []
        segment_b = []
        song_names = []
        for cur_item in batch:
            # set starting points
            start_point_a = torch.randint(low=0, high=max_length-input_length_a, size=(1,))[0]
            start_point_b = torch.randint(low=0, high=max_length-input_length_b, size=(1,))[0]
            # segmentize inputs
            segmentized_a = cur_item[0][:, start_point_a : start_point_a+input_length_a]
            segmentized_b = cur_item[1][:, start_point_b : start_point_b+input_length_b]
            segment_a.append(segmentized_a)
            segment_b.append(segmentized_b)
            song_names.append(cur_item[2])

        # returns segmentized inputs
        return torch.stack(segment_a, dim=0), torch.stack(segment_b, dim=0), song_names



'''
    Data Loaders
'''

# Data loader for training the 'Mastering Cloner'
    # loads two segments (A and B) from the same song
    # both segments are manipulated via Mastering Effects Manipulator (resulting A1, A2, and B2)
    # one of the manipulated segment is used as a reference segment (B2), which is randomly manipulated the same as the ground truth segment (A2)
class Song_Dataset_Mastering_Manipulated(Dataset):
    def __init__(self, args, mode):
        self.data_dir = args.data_dir + mode + "/"
        self.mode = mode
        self.segment_length = args.segment_length
        self.segment_length_ref = args.max_segment_length_feat
        self.fixed_random_seed = args.random_seed

        self.data_paths = glob(f"{self.data_dir}**/*.wav", recursive=True)

        self.manipulator = Mastering_Effects_Manipulator(block_size=2**10)


    def __len__(self):
        if self.mode=='train':
            return len(self.data_paths)
        elif self.mode=='valid':
            return len(self.data_paths) * 10


    def __getitem__(self, idx):
        if self.mode=="train":
            torch.manual_seed(int(time.time())*(idx+1) % (2**32-1))
            np.random.seed(int(time.time())*(idx+1) % (2**32-1))
            random.seed(int(time.time())*(idx+1) % (2**32-1))
        else:
            # fixed random seed for evaluation
            torch.manual_seed(idx*self.fixed_random_seed)
            np.random.seed(idx*self.fixed_random_seed)
            random.seed(idx*self.fixed_random_seed)

        # current audio path
        idx = idx%len(self.data_paths) if self.mode!="train" else idx
        cur_aud_path = self.data_paths[idx]
        
        # get 2 starting points of each segment
        start_point_a = torch.randint(low=0, high=load_wav_length(cur_aud_path)-self.segment_length, size=(1,))[0]
        start_point_b = torch.randint(low=0, high=load_wav_length(cur_aud_path)-self.segment_length_ref, size=(1,))[0]

        # load wav segments
        segment_a = load_wav_segment(cur_aud_path, start_point=start_point_a, duration=self.segment_length)
        segment_b = load_wav_segment(cur_aud_path, start_point=start_point_b, duration=self.segment_length_ref)

        # mastering manipulation
        manipulated_a1 = self.manipulator.process([segment_a], randomize=True, reset=False)[0]
        manipulated_a2, manipulated_b2 = self.manipulator.process([segment_a, segment_b], randomize=True, reset=False)

        manipulated_segment_a1 = torch.clamp(torch.transpose(torch.from_numpy(manipulated_a1).float(), 1, 0), min=-1, max=1)
        manipulated_segment_a2 = torch.clamp(torch.transpose(torch.from_numpy(manipulated_a2).float(), 1, 0), min=-1, max=1)
        manipulated_segment_b2 = torch.clamp(torch.transpose(torch.from_numpy(manipulated_b2).float(), 1, 0), min=-1, max=1)
        segment_a = torch.clamp(torch.transpose(torch.from_numpy(segment_a).float(), 1, 0), min=-1, max=1)

        return manipulated_segment_a1, manipulated_segment_a2, manipulated_segment_b2, segment_a



# Data loader for training the 'Music Effects Encoder'
    # loads two segments from the same song
class Song_Dataset(Dataset):
    def __init__(self, args, mode):
        self.data_dir = args.data_dir_feat if mode=='train' else args.data_dir_feat_test
        self.mode = mode
        self.segment_length = args.max_segment_length_feat
        self.args = args
        self.fixed_random_seed = args.random_seed
        self.data_paths = glob(f"{self.data_dir}**/*.wav", recursive=True)


    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, idx):
        if self.mode=='train':
            torch.manual_seed(int(time.time())*(idx+1) % (2**32-1))
            np.random.seed(int(time.time())*(idx+1) % (2**32-1))
            random.seed(int(time.time())*(idx+1) % (2**32-1))
        elif self.mode=='valid':
            # fixed random seed for evaluation
            torch.manual_seed(idx*self.fixed_random_seed)
            np.random.seed(idx*self.fixed_random_seed)
            random.seed(idx*self.fixed_random_seed)

        # current audio path
        cur_aud_path = self.data_paths[idx]
        song_name = cur_aud_path.split('/')[-1][:-4]
        
        if self.mode!='test':
            # get 2 starting points of each segment
            start_points = torch.randint(low=0, high=load_wav_length(cur_aud_path)-self.segment_length, size=(2,))

            # load wav segments
            segment_a = load_wav_segment(cur_aud_path, start_point=start_points[0], duration=self.segment_length).transpose(-1, -2)
            segment_b = load_wav_segment(cur_aud_path, start_point=start_points[1], duration=self.segment_length).transpose(-1, -2)
            segment_a = torch.from_numpy(segment_a).float()
            segment_b = torch.from_numpy(segment_b).float()

            return segment_a, segment_b, song_name
        
        else:
            # load whole audio file
            whole_wav = load_wav_segment(cur_aud_path, sample_rate=self.args.sample_rate).transpose(-1, -2)
            whole_wav = torch.from_numpy(whole_wav).float()

            return whole_wav, song_name



# Data loader for inferencing the task 'End-to-end Music Remastering'
    # loads whole songs of target and reference track
class Song_Dataset_Inference(Dataset):
    def __init__(self, args):
        self.data_dir = args.data_dir_test
        '''
            target files should be organized under the test data directory as follow
                "path_to_data_directory"/set_1/input.wav
                "path_to_data_directory"/set_1/reference.wav
                ...
                "path_to_data_directory"/set_n/input.wav
                "path_to_data_directory"/set_n/reference.wav
        '''
        self.data_paths_input = sorted(glob(f"{self.data_dir}*/input.wav"))
        self.data_paths_ref = sorted(glob(f"{self.data_dir}*/reference.wav"))
        assert len(self.data_paths_input)==len(self.data_paths_ref), "input files for the inference procedure mismatch"


    def __len__(self):
        return len(self.data_paths_input)


    def __getitem__(self, idx):
        # current audio path
        cur_aud_path_ori = self.data_paths_input[idx]
        cur_aud_path_ref = self.data_paths_ref[idx]

        ''' Loading wav files '''
        # load whole songs
        whole_song_ori = load_wav_segment(cur_aud_path_ori, axis=0)
        whole_song_ref = load_wav_segment(cur_aud_path_ref, axis=0)

        whole_song_ori = torch.from_numpy(whole_song_ori).float()
        whole_song_ref = torch.from_numpy(whole_song_ref).float()
        song_name = cur_aud_path_ori.split('/')[-2]

        return whole_song_ori, whole_song_ref, song_name





# check dataset
if __name__ == '__main__':
    """
    Test code of data loaders
    """
    from config import args
    print('--- checking dataset... ---')

    total_epochs = 2
    bs = 4
    step_size = 10
    collate_class = Collate_Variable_Length_Segments(args)

    
    ### Music Effects Encoder loaders
    print('\n========== Music Effects Encoder ==========')
    dataset = Song_Dataset(args, mode='train')
    train_loader = DataLoader(dataset, batch_size=bs, collate_fn=collate_class.manipulate_duration_both_segments, shuffle=False, drop_last=False, num_workers=0)

    for epoch in range(total_epochs):
        start_time_loader = time.time()
        for step, (segment_a, segment_b, song_name) in zip(range(step_size), train_loader):
            print(f'Epoch {epoch+1}/{total_epochs}\tStep {step+1}/{len(train_loader)}')
            print(segment_a.shape, segment_b.shape, f"time taken: {time.time()-start_time_loader:.4f}")
            start_time_loader = time.time()
    
    
    ### Mastering Cloner loaders
    print('\n========== Mastering Cloner ==========')    
    dataset = Song_Dataset_Mastering_Manipulated(args, mode='train')
    train_loader = DataLoader(dataset, batch_size=bs, collate_fn=collate_class.manipulate_duration_single_segment, shuffle=False, drop_last=False, num_workers=0)

    for epoch in range(total_epochs):
        start_time_loader = time.time()
        for step, (segment_a1, segment_a2, segment_b2, segment_a_ori) in zip(range(step_size), train_loader):
            print(f'Epoch {epoch+1}/{total_epochs}\tStep {step+1}/{len(train_loader)}')
            print(segment_a1.shape, segment_a2.shape, segment_b2.shape, f"time taken: {time.time()-start_time_loader:.4f}")
            start_time_loader = time.time()
    

    ### Mastering Cloner inference loaders
    print('\n========== Mastering Cloner Inference ==========')
    dataset = Song_Dataset_Inference(args)
    # batch size should be 1 to load full tracks
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    for epoch in range(total_epochs):
        start_time_loader = time.time()
        for step, (input_track, reference_track, song_name) in enumerate(train_loader):
            print(f'Epoch {epoch+1}/{total_epochs}\tStep {step+1}/{len(train_loader)}')
            print(input_track.shape, reference_track.shape, f"time taken: {time.time()-start_time_loader:.4f}")
            start_time_loader = time.time()


    print('\n--- checking dataset completed ---')