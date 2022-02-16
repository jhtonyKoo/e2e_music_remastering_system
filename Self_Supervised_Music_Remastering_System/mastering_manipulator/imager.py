"""
    Implementation of the processor 'multiband stereo imager'
"""
from scipy import signal
import numpy as np

# refer to pymixconsole at https://github.com/csteinmetz1/pymixconsole
from pymixconsole.processor import Processor
from pymixconsole.parameter import Parameter
from pymixconsole.parameter_list import ParameterList



class CrossoverMidSideImager(Processor):
    def __init__(self, crossover_order, name='IMAGER', block_size=512*256, sample_rate=44100):
        super().__init__(name, None, block_size, sample_rate)
        self.sample_rate = sample_rate
        self.crossover_order = crossover_order
        self.num_bands = 4

        # 4-band stereo imager
        self.parameters = ParameterList()
        self.parameters.add(Parameter("fq_1",   50.0,   "int",   processor=self, minimum=20.0,   maximum=1500.0))
        self.parameters.add(Parameter("fq_2",   1500.0, "int",   processor=self, minimum=1500.0, maximum=7000.0))
        self.parameters.add(Parameter("fq_3",   7000.0, "int",   processor=self, minimum=7000.0, maximum=20000.0))
        self.parameters.add(Parameter("bal_1",  0.0,    "float", processor=self, minimum=0.0,    maximum=3.0))
        self.parameters.add(Parameter("bal_2",  0.0,    "float", processor=self, minimum=0.0,    maximum=3.0))
        self.parameters.add(Parameter("bal_3",  0.0,    "float", processor=self, minimum=0.0,    maximum=3.0))
        self.parameters.add(Parameter("bal_4",  0.0,    "float", processor=self, minimum=0.0,    maximum=3.0))


    def process(self, data):
        # input shape : [signal length, 2]

        # retrieve values of crossover frequency bands and their midside balance
        fq_vals = []
        bals_vals = []
        for i in range(self.num_bands-1):
            fq_vals.append(round(getattr(self.parameters, f"fq_{i+1}").value, 1))
        for k in range(self.num_bands):
            bals_vals.append(round(getattr(self.parameters, f"bal_{k+1}").value, 2))

        # apply multiband stereo imager
        imaged = self.image(data.transpose(), np.array(fq_vals), np.array(bals_vals))
        return imaged.transpose()


    def image(self,
              input_audio,                   # shape: (2, signal_length)
              crossover_frequencies,         # shape: (crossover_order,), value: 0 ~ sampling rate.
              midside_balance):              # shape: (crossover_order + 1,), value: 0 (centered) ~.
        # apply crossover filter
        crossovered = self.crossover(input_audio, crossover_frequencies)

        # manipulate multiband stereo images
        ms_processed = []
        for i in range(self.num_bands):
            band_mid, band_side = self.lr_to_ms(crossovered[i][0], crossovered[i][1])
            band_mid_e, band_side_e = np.sum(band_mid ** 2), np.sum(band_side ** 2)
            total_e = band_mid_e + band_side_e
            max_side_multiplier = np.sqrt(total_e / (band_side_e + 1e-3))
            side_gain = min(midside_balance[i], max_side_multiplier)
            new_band_side = band_side * side_gain
            new_band_side_e = band_side_e * (side_gain ** 2)
            left_mid_e = total_e - new_band_side_e
            mid_gain = np.sqrt(left_mid_e / (band_mid_e + 1e-3))
            new_band_mid = band_mid * mid_gain
            band_left, band_right = self.ms_to_lr(new_band_mid, new_band_side)
            ms_processed.append(np.stack([band_left, band_right], 0))

        ms_processed = sum(ms_processed)

        return ms_processed


    # Linkwitz-Riley crossover filter
    def crossover(self, input_audio, crossover_frequencies):
        crossover_frequencies = np.sort(crossover_frequencies)
        num_crossovers = len(crossover_frequencies)

        lpf_soses, hpf_soses = [], []
        for crossover_frequency in crossover_frequencies:
            lpf_sos = signal.butter(self.crossover_order, 
                                    crossover_frequency,
                                    'lowpass',
                                    fs = self.sample_rate,
                                    output = 'sos')
            lpf_soses.append(lpf_sos)
            hpf_sos = signal.butter(self.crossover_order, 
                                    crossover_frequency,
                                    'highpass',
                                    fs = self.sample_rate,
                                    output = 'sos')
            hpf_soses.append(hpf_sos)

        filtered_signals = []
        # apply forward-backward digital filter (zero-phase filter impulse response)
        for i in range(num_crossovers):
            lpfed = signal.sosfiltfilt(lpf_soses[i], input_audio)
            filtered_signals.append(lpfed)
            hpfed = signal.sosfiltfilt(hpf_soses[i], input_audio)
            if self.crossover_order % 2 == 1:
                hpfed = hpfed * -1.
            if i == num_crossovers - 1:
                filtered_signals.append(hpfed)
            input_audio = hpfed

        return filtered_signals


    # left-right channeled signal to mid-side signal
    def lr_to_ms(self, left, right):
        mid = left + right
        side = left - right
        return mid, side


    # mid-side channeled signal to left-right signal
    def ms_to_lr(self, mid, side):
        left = (mid + side) / 2
        right = (mid - side) / 2
        return left, right


    def update(self, parameter_name=None):
        return parameter_name



if __name__ == '__main__':
    ''' check I/O shape '''
    print("---Checking I/O shape of the module 'Imager'---")
    sr = 44100
    segment_length = sr*10

    imager = CrossoverMidSideImager(crossover_order = 4)
    imager.randomize()

    in_sig = np.random.rand(segment_length, 2)
    out_sig = imager.process(in_sig)
    print(f"input shape : {in_sig.shape}\noutput shape : {out_sig.shape}")
