"""
    Implementation of the processor 'gain' from "pymixconsole"
        available at https://github.com/csteinmetz1/pymixconsole/blob/master/pymixconsole/processors/gain.py
"""
from numba import jit, float64

# refer to pymixconsole at https://github.com/csteinmetz1/pymixconsole
from pymixconsole.processor import Processor
from pymixconsole.parameter import Parameter
from pymixconsole.parameter_list import ParameterList


@jit(nopython=True)
def n_process(data, gain):
    return gain * data

class Gain(Processor):
    def __init__(self, gain, name="PRE GAIN", parameters=None, block_size=512, sample_rate=44100):
        super().__init__(name, parameters, block_size, sample_rate)
        self.gain = gain

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter("gain", self.gain, "float", processor=None, units="dB", minimum=self.gain, maximum=self.gain))

    def process(self, data):
        data = n_process(data, self.db2linear(self.parameters.gain.value))
        return data

    def update(self, parameter_name):
        pass