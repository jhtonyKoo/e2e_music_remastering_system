"""
    Implementation of the processor 'compressor' from "pymixconsole"
        available at https://github.com/csteinmetz1/pymixconsole/blob/master/pymixconsole/processors/compressor.py
"""
from numba import jit
import numpy as np

# refer to pymixconsole at https://github.com/csteinmetz1/pymixconsole
from pymixconsole.processor import Processor
from pymixconsole.parameter import Parameter
from pymixconsole.parameter_list import ParameterList

@jit(nopython=True)
def n_process(data, buffer, M, threshold, attack_time, release_time, ratio, makeup_gain, sample_rate, yL_prev):
    x_g = np.zeros(M)
    x_l = np.zeros(M)
    y_g = np.zeros(M)
    y_l = np.zeros(M)
    c   = np.zeros(M)

    alpha_attack  = np.exp(-1/(0.001 * sample_rate * attack_time))
    alpha_release = np.exp(-1/(0.001 * sample_rate * release_time))

    for i in np.arange(M):
        if np.abs(buffer[i]) <  0.000001:
            x_g[i] = -120.0
        else:
            x_g[i] = 20 * np.log10(np.abs(buffer[i]))

        if x_g[i] >= threshold:
            y_g[i] = threshold + (x_g[i] - threshold) / ratio
        else:
            y_g[i] = x_g[i]
            
        x_l[i] = x_g[i] - y_g[i]

        if x_l[0] > yL_prev:
            y_l[i] = alpha_attack * yL_prev + (1 - alpha_attack ) * x_l[i]
        else:
            y_l[i] = alpha_release * yL_prev + (1 - alpha_release) * x_l[i]

        c[i] = np.power(10.0, (makeup_gain - y_l[i]) / 20.0)
        yL_prev = y_l[i]

    if False:
        data[:,0] *= c
        data[:,1] *= c
    else:
        data *= c

    return data, yL_prev



# limiter type 1 / From pymix Compressor
class LimiterTypeA(Processor):
    """ Single band dynamic range compressor.
    """
    def __init__(self, threshold=-7.0, ratio=6.0, gain=7.0, name="LIMITER TYPE1", block_size=512, sample_rate=44100):
        super().__init__(name, None, block_size, sample_rate)
        self.parameters = ParameterList()
        self.parameters.add(Parameter("threshold",    threshold, "int",   units="dB", processor=self, minimum=-10.0, maximum=-6.0))
        self.parameters.add(Parameter("attack_time",  0.5,       "float", units="ms", processor=self, minimum=0.1,   maximum=30.0))
        self.parameters.add(Parameter("release_time", 50.0,      "int",   units="ms", processor=self, minimum=50.0,  maximum=100.0))
        self.parameters.add(Parameter("ratio",        ratio,     "int",               processor=self, minimum=5.0,   maximum=10.0))
        self.parameters.add(Parameter("makeup_gain",  gain,      "int",   units="dB", processor=self, minimum=4.0,   maximum=12.0))

        self.yL_prev = 0

    def process(self, x):
        buffer = np.squeeze((x[:,0] + x[:,1])) * 0.5

        x, self.yL_prev = n_process(x,
                    buffer, 
                    x.shape[0],
                    self.parameters.threshold.value, 
                    self.parameters.attack_time.value,
                    self.parameters.release_time.value,
                    self.parameters.ratio.value,
                    self.parameters.makeup_gain.value,
                    self.sample_rate,
                    self.yL_prev)

        self.yL_prev = 0
        return x

    def update(self, parameter_name):
        pass


if __name__ == "__main__":
    ''' check I/O shape '''
    print("---Checking I/O shape of the module 'Limiter'---")
    sr = 44100
    segment_length = sr*10

    imager = LimiterTypeA()
    imager.randomize()

    in_sig = np.random.rand(segment_length, 2)
    out_sig = imager.process(in_sig)
    print(f"input shape : {in_sig.shape}\noutput shape : {out_sig.shape}")
