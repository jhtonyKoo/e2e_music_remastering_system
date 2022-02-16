"""
    Utility File
        containing functions for neural networks
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import torchaudio



# 2-dimensional convolutional layer
# in the order of conv -> norm -> activation
class Conv2d_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, \
                                    stride=1, \
                                    padding="SAME", dilation=(1,1), bias=True, \
                                    norm="batch", activation="relu", \
                                    mode="conv"):
        super(Conv2d_layer, self).__init__()

        self.conv2d = nn.Sequential()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        if isinstance(stride, int):
            stride = [stride, stride]
        if isinstance(dilation, int):
            dilation = [dilation, dilation]

        ''' padding '''
        if mode=="deconv":
            padding = tuple(int((current_kernel - 1)/2) for current_kernel in kernel_size)
            out_padding = tuple(0 if current_stride == 1 else 1 for current_stride in stride)
        elif mode=="conv":
            if padding == "SAME":
                f_pad = int((kernel_size[0]-1) * dilation[0])
                t_pad = int((kernel_size[1]-1) * dilation[1])
                t_l_pad = int(t_pad//2)
                t_r_pad = t_pad - t_l_pad
                f_l_pad = int(f_pad//2)
                f_r_pad = f_pad - f_l_pad
                padding_area = (t_l_pad, t_r_pad, f_l_pad, f_r_pad)
            elif padding == "VALID":
                padding = 0
            else:
                pass

        ''' convolutional layer '''
        if mode=="deconv":
            self.conv2d.add_module("deconv2d", nn.ConvTranspose2d(in_channels, out_channels, \
                                                            (kernel_size[0], kernel_size[1]), \
                                                            stride=stride, \
                                                            padding=padding, output_padding=out_padding, \
                                                            dilation=dilation, \
                                                            bias=bias))
        elif mode=="conv":
            self.conv2d.add_module(f"{mode}2d_pad", nn.ReflectionPad2d(padding_area))
            self.conv2d.add_module(f"{mode}2d", nn.Conv2d(in_channels, out_channels, \
                                                            (kernel_size[0], kernel_size[1]), \
                                                            stride=stride, \
                                                            padding=0, \
                                                            dilation=dilation, \
                                                            bias=bias))
        
        ''' normalization '''
        if norm=="batch":
            self.conv2d.add_module("batch_norm", nn.BatchNorm2d(out_channels))

        ''' activation '''
        if activation=="relu":
            self.conv2d.add_module("relu", nn.ReLU())
        elif activation=="lrelu":
            self.conv2d.add_module("lrelu", nn.LeakyReLU())


    def forward(self, input):
        # input shape should be : batch x channel x height x width
        output = self.conv2d(input)
        return output



# 1-dimensional convolutional layer
# in the order of conv -> norm -> activation
class Conv1d_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, \
                                    stride=1, \
                                    padding="SAME", dilation=1, bias=True, \
                                    norm="batch", activation="relu", \
                                    mode="conv"):
        super(Conv1d_layer, self).__init__()
        
        self.conv1d = nn.Sequential()

        ''' padding '''
        if mode=="deconv":
            padding = int(dilation * (kernel_size-1) / 2)
            out_padding = 0 if stride==1 else 1
        elif mode=="conv" or "alias_free" in mode:
            if padding == "SAME":
                pad = int((kernel_size-1) * dilation)
                l_pad = int(pad//2)
                r_pad = pad - l_pad
                padding_area = (l_pad, r_pad)
            elif padding == "VALID":
                padding_area = (0, 0)
            else:
                pass

        ''' convolutional layer '''
        if mode=="deconv":
            self.conv1d.add_module("deconv1d", nn.ConvTranspose1d(in_channels, out_channels, kernel_size, \
                                                            stride=stride, padding=padding, output_padding=out_padding, \
                                                            dilation=dilation, \
                                                            bias=bias))
        elif mode=="conv":
            self.conv1d.add_module(f"{mode}1d_pad", nn.ReflectionPad1d(padding_area))
            self.conv1d.add_module(f"{mode}1d", nn.Conv1d(in_channels, out_channels, kernel_size, \
                                                            stride=stride, padding=0, \
                                                            dilation=dilation, \
                                                            bias=bias))
        elif "alias_free" in mode:
            if "up" in mode:
                up_factor = stride * 2
                down_factor = 2
            elif "down" in mode:
                up_factor = 2
                down_factor = stride * 2
            else:
                raise ValueError("choose alias-free method : 'up' or 'down'")
            # procedure : conv -> upsample -> lrelu -> low-pass filter -> downsample
            # the torchaudio.transforms.Resample's default resampling_method is 'sinc_interpolation' which performs low-pass filter during the process
            # details at https://pytorch.org/audio/stable/transforms.html
            self.conv1d.add_module(f"{mode}1d_pad", nn.ReflectionPad1d(padding_area))
            self.conv1d.add_module(f"{mode}1d", nn.Conv1d(in_channels, out_channels, kernel_size, \
                                                            stride=1, padding=0, \
                                                            dilation=dilation, \
                                                            bias=bias))
            self.conv1d.add_module(f"{mode}upsample", torchaudio.transforms.Resample(orig_freq=1, new_freq=up_factor))
            self.conv1d.add_module(f"{mode}lrelu", nn.LeakyReLU())
            self.conv1d.add_module(f"{mode}downsample", torchaudio.transforms.Resample(orig_freq=down_factor, new_freq=1))

        ''' normalization '''
        if norm=="batch":
            self.conv1d.add_module("batch_norm", nn.BatchNorm1d(out_channels))

        ''' activation '''
        if 'alias_free' not in mode:
            if activation=="relu":
                self.conv1d.add_module("relu", nn.ReLU())
            elif activation=="lrelu":
                self.conv1d.add_module("lrelu", nn.LeakyReLU())


    def forward(self, input):
        # input shape should be : batch x channel x height x width
        output = self.conv1d(input)
        return output



# Residual Block
    # the input is added after the first convolutional layer, retaining its original channel size
    # therefore, the second convolutional layer's output channel may differ
class Res_ConvBlock(nn.Module):
    def __init__(self, dimension, \
                        in_channels, out_channels, \
                        kernel_size, \
                        stride=1, padding="SAME", \
                        dilation=1, \
                        bias=True, \
                        norm="batch", \
                        activation="relu", last_activation="relu", \
                        mode="conv"):
        super(Res_ConvBlock, self).__init__()

        if dimension==1:
            self.conv1 = Conv1d_layer(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=activation)
            self.conv2 = Conv1d_layer(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=last_activation, mode=mode)
        elif dimension==2:
            self.conv1 = Conv2d_layer(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=activation)
            self.conv2 = Conv2d_layer(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=last_activation, mode=mode)


    def forward(self, input):
        c1_out = self.conv1(input) + input
        c2_out = self.conv2(c1_out)
        return c2_out



# Convoluaionl Block
    # consists of multiple (number of layer_num) convolutional layers
    # only the final convoluational layer outputs the desired 'out_channels'
class ConvBlock(nn.Module):
    def __init__(self, dimension, layer_num, \
                        in_channels, out_channels, \
                        kernel_size, \
                        stride=1, padding="SAME", \
                        dilation=1, \
                        bias=True, \
                        norm="batch", \
                        activation="relu", last_activation="relu", \
                        mode="conv"):
        super(ConvBlock, self).__init__()

        conv_block = []
        if dimension==1:
            for i in range(layer_num-1):
                conv_block.append(Conv1d_layer(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=activation))
            conv_block.append(Conv1d_layer(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=last_activation, mode=mode))
        elif dimension==2:
            for i in range(layer_num-1):
                conv_block.append(Conv2d_layer(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=activation))
            conv_block.append(Conv2d_layer(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=last_activation, mode=mode))
        self.conv_block = nn.Sequential(*conv_block)


    def forward(self, input):
        return self.conv_block(input)



# Feature-wise Linear Modulation
class FiLM(nn.Module):
    def __init__(self, condition_len=2048, feature_len=1024):
        super(FiLM, self).__init__()
        self.film_fc = nn.Linear(condition_len, feature_len*2)
        self.feat_len = feature_len


    def forward(self, feature, condition):
        film_factor = self.film_fc(condition).unsqueeze(-1)
        r, b = torch.split(film_factor, self.feat_len, dim=1)
        return r*feature + b

