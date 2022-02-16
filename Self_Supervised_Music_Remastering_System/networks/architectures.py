""" 
    Implementation of neural networks used in the task 'End-to-end Remastering System'
        - 'Mastering Cloner', '2D Projection Discriminator', and 'Music Effects Encoder'
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import os
import sys
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)

# from network.utils import *
from Self_Supervised_Music_Remastering_System.networks.utils import *



# Mastering Cloner
    # Wave-U-net based with FiLM conditioning method
class Mastering_Cloner_Wave_Unet(nn.Module):
    def __init__(self, config, verbose=False):
        super(Mastering_Cloner_Wave_Unet, self).__init__()
        self.verbose = verbose

        ''' conditioning layers '''
        self.cond_layers = config["cond_layers"]
        self.cond_place = config["cond_place"]
        self.conditioners = {}
        conditioners = []
        if self.cond_layers=="ALL":
            for cur_layer, cur_channel in enumerate(config["channels"]):
                conditioners.append(FiLM(condition_len=config["condition_dimension"], feature_len=cur_channel))
        else:
            self.cond_layers = [cur_idx % len(config["channels"]) for cur_idx in self.cond_layers]  # convert negative index to positive
            if self.cond_place=="dec":
                self.cond_layers = [len(config["channels"])-1-cur_idx for cur_idx in self.cond_layers]
            self.cond_layers.sort()
            for cur_layer in self.cond_layers:
                conditioners.append(FiLM(condition_len=config["condition_dimension"], feature_len=config["channels"][cur_layer]))
        if self.cond_place=="dec":
            conditioners.reverse()
            if self.cond_layers=="ALL":
                conditioners.pop(0)
        self.conditioners = nn.Sequential(*conditioners)

        ''' encoder '''
        # input / output is stereo channeled audio
        config["channels"].insert(0, 2)

        # encoder layers
        encoder = []
        mode = config["downconv_method"] if "downconv_method" in config else "conv"
        for i in range(len(config["kernels"])):
            cur_activation = config["first_activation"] if "first_activation" in config and i==0 else config["activation"]
            encoder.append(ConvBlock(dimension=1, layer_num=config["conv_layer_num"], \
                                        in_channels=config["channels"][i], out_channels=config["channels"][i+1], \
                                        kernel_size=config["kernels"][i], \
                                        stride=config["strides"][i], padding="SAME", \
                                        dilation=config["dilation"][i], \
                                        norm=config["norm"], \
                                        activation=cur_activation, \
                                        mode=mode))
        self.encoder = nn.Sequential(*encoder)

        ''' decoder '''
        # decoder configs - considering skip connections
        config["kernels"].reverse()
        config["strides"].reverse()
        config["dilation"].reverse()
        config["channels"].reverse()
        in_channels = config["channels"].copy()
        for i in range(1, len(self.encoder)):
            in_channels[i] += config["channels"][i]
        
        self.apply_last_conv_layers = config["last_conv_layers"]
        if self.apply_last_conv_layers:
            config["channels"].insert(-2, config["channels"][-2])

        # decoder layers
        decoder = []
        last_activation = config["activation"]
        for i in range(len(config["kernels"])):
            if i==len(config["kernels"])-1 and not self.apply_last_conv_layers: last_activation = config["last_activation"]
            mode = config["deconv_method"] if i < len(config["kernels"]) else "conv"
            decoder.append(ConvBlock(dimension=1, layer_num=config["conv_layer_num"], \
                                        in_channels=in_channels[i], out_channels=config["channels"][i+1], \
                                        kernel_size=config["kernels"][i], \
                                        stride=config["strides"][i], padding="SAME", \
                                        dilation=config["dilation"][i], \
                                        norm=config["norm"], \
                                        activation=config["activation"], last_activation=last_activation, \
                                        mode=mode))
        self.decoder = nn.Sequential(*decoder)

        # last conv layer
        if self.apply_last_conv_layers:
            self.last_conv_layers = ConvBlock(dimension=1, layer_num=config["conv_layer_num"], \
                                                in_channels=config["channels"][-2], out_channels=config["channels"][-1], \
                                                kernel_size=config["kernels"][-1], \
                                                stride=1, \
                                                norm=config["norm"], \
                                                activation=config["activation"], last_activation=config["last_activation"], \
                                                mode="conv")


    # network forward operation
    def forward(self, src_audio, ref_embedding):
        # receive raw source audio and reference embedding
        # outputs referenced styled matered output

        ''' encode '''
        condition_embedding = ref_embedding if "enc" in self.cond_place else None
        enc_latent = self.encoding(src_audio, condition=condition_embedding, verbose=self.verbose)

        ''' decode '''
        condition_embedding = ref_embedding if "dec" in self.cond_place else None
        # skip connect raw channels
        output = self.decoding(enc_latent[-1], enc_latent[:-1], src_audio, condition=condition_embedding, verbose=self.verbose)

        if self.apply_last_conv_layers:
            output = self.last_conv_layers(output)

        output = torch.clamp(output, min=-1, max=1)

        return output


    # encoding procedure
    def encoding(self, input, condition=None, verbose=False):
        enc_latent = []
        current_input = input
        cond_idx = 0
        
        if verbose:
            print('---encoding---')
            print(current_input.shape)

        for block_num, current_conv_block in enumerate(self.encoder):
            current_input = current_conv_block(current_input)
            # conditioning
            if condition!=None and (self.cond_layers=="ALL" or block_num in self.cond_layers):
                current_input = self.conditioners[cond_idx](current_input, condition)
                cond_idx += 1
                if verbose:
                    print(f'conditioning at layer {block_num}')
            enc_latent.append(current_input)

            if verbose:
                print(current_input.shape)

        return enc_latent
    

    # decoding procedure
    def decoding(self, input, skip_latent, last_skip, condition=None, verbose=False):
        skip_latent.reverse()
        current_input = input
        cond_idx = 0

        if verbose:
            print('---decoding---')
            print(current_input.shape)

        for block_num, current_conv_block in enumerate(self.decoder):
            # decoding block
            current_input = current_conv_block(current_input)
            # conditioning & skip connect for next layer
            if block_num is not len(self.decoder)-1:
                # conditioning
                if condition!=None and (self.cond_layers=="ALL" or block_num in self.cond_layers):
                    current_input = self.conditioners[cond_idx](current_input, condition)
                    cond_idx += 1
                    if verbose:
                        print(f'conditioning at layer {block_num}')
                # skip connection
                skip_diff_amount = current_input.shape[-1] - skip_latent[block_num].shape[-1]
                if skip_diff_amount==1:
                    current_input = current_input[..., :-1]
                elif skip_diff_amount==2:
                    current_input = current_input[..., 1:-1]
                current_input = torch.cat((current_input, skip_latent[block_num]), dim=1)

            if verbose:
                print(current_input.shape)

        return current_input



# Mastering Cloning Discriminator 2D
    # with projection of encoded embedding from the Music Effects Encoder
class Projection_Discriminator_2D(nn.Module):
    def __init__(self, config):
        super(Projection_Discriminator_2D, self).__init__()

        # input is stereo channeled magnitude spectrogram
        config["channels"].insert(0, 2)
        # encoder
        encoder = []
        for i in range(len(config["kernels"])):
            encoder.append(ConvBlock(dimension=2, layer_num=config["conv_layer_num"], \
                                        in_channels=config["channels"][i], out_channels=config["channels"][i+1], \
                                        kernel_size=config["kernels"][i], \
                                        stride=config["strides"][i], padding="SAME", \
                                        dilation=config["dilation"][i], \
                                        norm=config["norm"], \
                                        activation=config["activation"]))
        self.encoder = nn.Sequential(*encoder)

        self.glob_pool = nn.AdaptiveAvgPool2d(1)

        self.out_fc = nn.utils.spectral_norm(
            nn.Linear(config["channels"][-1], 1)
        )
        self.cond_mlp = nn.Sequential(
                nn.Linear(config["condition_dimension"], 1024),
                nn.utils.spectral_norm(
                    nn.Linear(1024, config["channels"][-1])
                )
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    
    # network forward operation
    def forward(self, src_audio, ref_embedding):
        # receive raw source audio and reference embedding
        # discriminates input audio using reference embedding

        ''' encode '''
        enc_output = self.encoder(src_audio)
        # global pooling
        pooled = self.glob_pool(enc_output).squeeze(-1).squeeze(-1)

        # projection discriminator based method
        output = self.out_fc(pooled)
        ref_embedding_out = self.cond_mlp(ref_embedding)

        c = torch.sum(pooled*ref_embedding_out, dim=1, keepdim=True)

        return self.tanh(output+c)



# Encoder of music effects for contrastive learning of music effects
class Music_Effects_Encoder(nn.Module):
    def __init__(self, config, reload_ckpt=True):
        super(Music_Effects_Encoder, self).__init__()
        self.reload_ckpt = reload_ckpt

        # input is stereo channeled audio
        config["channels"].insert(0, 2)

        # encoder layers
        encoder = []
        for i in range(len(config["kernels"])):
            encoder.append(Res_ConvBlock(dimension=1, in_channels=config["channels"][i], out_channels=config["channels"][i+1], \
                                            kernel_size=config["kernels"][i], \
                                            stride=config["strides"][i], padding="SAME", \
                                            dilation=config["dilation"][i], \
                                            norm=config["norm"], \
                                            activation=config["activation"], last_activation=config["activation"]))
        self.encoder = nn.Sequential(*encoder)

        # pooling method
        self.glob_pool = nn.AdaptiveAvgPool1d(1)

        # last FC layer
        if reload_ckpt:
            self.fc = nn.Linear(config["channels"][-1], config["z_dim"])


    # network forward operation
    def forward(self, input):
        enc_output = self.encoder(input)
        glob_pooled = self.glob_pool(enc_output).squeeze(-1)

        if self.reload_ckpt:
            output = self.fc(glob_pooled)
        else:
            output = None

        # outputs z and c feature
        return output, glob_pooled





if __name__ == '__main__':
    ''' check model I/O shape '''
    import yaml
    with open('configs.yaml', 'r') as f:
        configs = yaml.load(f)

    batch_size = 32
    sr = 44100
    input_length = sr*5
    
    input = torch.rand(batch_size, 2, input_length)
    print(f"Input Shape : {input.shape}\n")
    

    print('\n========== Music Effects Encoder ==========')
    model_arc = "Effects_Encoder"
    model_options = "default"

    config = configs[model_arc][model_options]
    print(f"configuration: {config}")

    network = Music_Effects_Encoder(config)
    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Number of trainable parameters : {pytorch_total_params}")

    # model inference
    output_z, output_c = network(input)
    print(f"Output Shape : z={output_z.shape}   c={output_c.shape}")
    


    print('\n========== Mastering Cloner ==========')    
    model_arc = "Mastering_Cloner"
    model_options = "default"

    config = configs[model_arc][model_options]
    print(f"configuration: {config}")

    network = Mastering_Cloner_Wave_Unet(config, verbose=True)
    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Number of trainable parameters : {pytorch_total_params}")

    # ref_embedding = torch.rand(batch_size, 2048)
    ref_embedding = output_c
    # model inference
    output = network(input, ref_embedding)
    print(f"Output Shape : {output.shape}")
    


    print('\n========== Projection_Discriminator ==========') 
    input = torch.rand(batch_size, 2, 1024, 256)
    print(f"Input Shape : {input.shape}")

    model_arc = "Projection_Discriminator_2D"
    model_options = "default"
    config = configs[model_arc][model_options]
    print(config)

    network = Projection_Discriminator_2D(config)
    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Number of trainable parameters : {pytorch_total_params}")

    # ref_embedding = torch.rand(batch_size, 2048)
    ref_embedding = output_c
    # model inference
    output = network(input, ref_embedding)
    print(f"Output Shape : {output.shape}\n")
    
