""" Front-end: processing raw data input """
import torch
import torch.nn as nn
import torchaudio.functional as ta_F



class FrontEnd(nn.Module):
    def __init__(self, channel='stereo', n_fft=2048, hop_length=None, win_length=None, window="hann", device=torch.device("cpu")):
        super(FrontEnd, self).__init__()
        self.channel = channel
        self.n_fft = n_fft
        self.hop_length = n_fft//4 if hop_length==None else hop_length
        self.win_length = n_fft if win_length==None else win_length
        if window=="hann":
            self.window = torch.hann_window(window_length=self.win_length, periodic=True).to(device)
        elif window=="hamming":
            self.window = torch.hamming_window(window_length=self.win_length, periodic=True).to(device)


    def forward(self, input, mode):
        # front-end function which channel-wise combines all demanded features
        # input shape : batch x channel x raw waveform
        # output shape : batch x channel x frequency x time

        front_output_list = []
        for cur_mode in mode:
            # Real & Imaginary
            if cur_mode=="cplx":
                if self.channel=="mono":
                    output = torch.stft(input, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                elif self.channel=="stereo":
                    output_l = torch.stft(input[:,0], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                    output_r = torch.stft(input[:,1], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                    output = torch.cat((output_l, output_r), axis=-1)
                if input.shape[2] % round(self.n_fft/4) == 0:
                    output = output[:, :, :-1]
                if self.n_fft % 2 == 0:
                    output = output[:, :-1]
                front_output_list.append(output.permute(0, 3, 1, 2))
            # Magnitude & Phase
            elif cur_mode=="mag":
                if self.channel=="mono":
                    cur_cplx = torch.stft(input, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                    output = self.mag(cur_cplx).unsqueeze(-1)[..., 0:1]
                elif self.channel=="stereo":
                    cplx_l = torch.stft(input[:,0], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                    cplx_r = torch.stft(input[:,1], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                    mag_l = self.mag(cplx_l).unsqueeze(-1)
                    mag_r = self.mag(cplx_r).unsqueeze(-1)
                    output = torch.cat((mag_l, mag_r), axis=-1)

                if input.shape[-1] % round(self.n_fft/4) == 0:
                    output = output[:, :, :-1]
                if self.n_fft % 2 == 0: # discard highest frequency
                    output = output[:, 1:]
                front_output_list.append(output.permute(0, 3, 1, 2))

        # combine all demanded features
        if not front_output_list:
            raise NameError("NameError at FrontEnd: check using features for front-end")
        elif len(mode)!=1:
            for i, cur_output in enumerate(front_output_list):
                if i==0:
                    front_output = cur_output
                else:
                    front_output = torch.cat((front_output, cur_output), axis=1)
        else:
            front_output = front_output_list[0]
            
        return front_output


    def mag(self, cplx_input, eps=1e-07):
        mag_summed = cplx_input.pow(2.).sum(-1) + eps
        return mag_summed.pow(0.5)





if __name__ == '__main__':

    batch_size = 16
    channel = 2
    segment_length = 512*128*6
    input_wav = torch.rand((batch_size, channel, segment_length))

    mode = ["cplx", "mag"]
    fe = FrontEnd(channel="stereo")
    
    output = fe(input_wav, mode=mode)
    print(f"Input shape : {input_wav.shape}\nOutput shape : {output.shape}")
