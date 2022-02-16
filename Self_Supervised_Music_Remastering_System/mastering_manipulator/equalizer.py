"""
    Implementation of the processor 'equaliser' from "pymixconsole"
        available at https://github.com/csteinmetz1/pymixconsole/blob/master/pymixconsole/processors/equaliser.py
"""
import numpy as np

# refer to pymixconsole at https://github.com/csteinmetz1/pymixconsole
from pymixconsole.processor import Processor
from pymixconsole.parameter import Parameter
from pymixconsole.parameter_list import ParameterList
from pymixconsole.components.iirfilter import IIRfilter



BANDS = ["low_shelf", "first_band", "second_band", "third_band", "high_shelf"]

class Equalizer(Processor):
    """ Five band parametreic equaliser ( two shelves and three central bands )
    All gains are set in dB values and range from `MIN_GAIN` dB to `MAX_GAIN` dB.
    This processor is implemented as cascade of five biquad IIR filters
    that are implemented using the infamous cookbook formulae from RBJ.
    """
    def __init__(self, name="EQUALIZER", block_size=512, sample_rate=44100, gain_range=(-10.0,5.0), q_range=(5.0, 30.0), hard_clip=False):
        super().__init__(name, None, block_size, sample_rate)

        MIN_GAIN = gain_range[0]
        MAX_GAIN = gain_range[1]
        MIN_Q    = q_range[0]
        MAX_Q    = q_range[1]

        self.parameters = ParameterList()
        # low shelf parameters ----------------------------------------------------------------------------------------------
        self.parameters.add(Parameter("low_shelf_gain",      0.0, "int", processor=self, minimum=MIN_GAIN, maximum=MAX_GAIN))
        self.parameters.add(Parameter("low_shelf_freq",     80.0, "int", processor=self, minimum=20.0,     maximum=1000.0))
        # first band parameters ---------------------------------------------------------------------------------------------
        self.parameters.add(Parameter("first_band_gain",     0.0, "int", processor=self, minimum=MIN_GAIN, maximum=MAX_GAIN))
        self.parameters.add(Parameter("first_band_freq",   400.0, "int", processor=self, minimum=200.0,    maximum=5000.0))        
        self.parameters.add(Parameter("first_band_q",        5.0, "int", processor=self, minimum=MIN_Q,    maximum=MAX_Q))
        # second band parameters --------------------------------------------------------------------------------------------
        self.parameters.add(Parameter("second_band_gain",    0.0, "int", processor=self, minimum=MIN_GAIN, maximum=MAX_GAIN))
        self.parameters.add(Parameter("second_band_freq", 1000.0, "int", processor=self, minimum=500.0,    maximum=6000.0))        
        self.parameters.add(Parameter("second_band_q",       5.0, "int", processor=self, minimum=MIN_Q,    maximum=MAX_Q))
        # third band parameters --------------------------------------------------------------------------------------------
        self.parameters.add(Parameter("third_band_gain",     0.0, "int", processor=self, minimum=MIN_GAIN, maximum=MAX_GAIN))
        self.parameters.add(Parameter("third_band_freq",  5000.0, "int", processor=self, minimum=2000.0,   maximum=10000.0))        
        self.parameters.add(Parameter("third_band_q",        5.0, "int", processor=self, minimum=MIN_Q,    maximum=MAX_Q))
        # high shelf parameters --------------------------------------------------------------------------------------------
        self.parameters.add(Parameter("high_shelf_gain",      0.0, "int", processor=self, minimum=MIN_GAIN,maximum=MAX_GAIN))
        self.parameters.add(Parameter("high_shelf_freq",  10000.0, "int", processor=self, minimum=8000.0,  maximum=20000.0))

        self.bands, self.filters = self.setup_filters()
        self.hard_clip = hard_clip


    def setup_filters(self):

        filters = {}

        for band in BANDS:

            G = getattr(self.parameters, band + "_gain").value
            fc = getattr(self.parameters, band + "_freq").value
            rate = self.sample_rate

            if band in ["low_shelf", "high_shelf"]:
                Q = 0.707
                filter_type = band
            else:
                Q = getattr(self.parameters, band + "_q").value
                filter_type = "peaking"

            filters[band] = IIRfilter(G, Q, fc, rate, filter_type, n_channels=2)

        return BANDS, filters


    def update_filter(self, band):

        self.filters[band].G    = getattr(self.parameters, band + "_gain").value
        self.filters[band].fc   = getattr(self.parameters, band + "_freq").value
        self.filters[band].rate = self.sample_rate 

        if band in ["first_band", "second_band", "third_band"]:
            self.filters[band].Q    = getattr(self.parameters, band + "_q").value


    def update(self, parameter_name):
        pass

        
    def reset_state(self):
        for band, iirfilter in self.filters.items():
            iirfilter.reset_state()


    def process(self, data):

        for band, irrfilter in self.filters.items():
            data = irrfilter.apply_filter(data)

        if self.hard_clip:
            data = np.clip(data, -1.0, 1.0)

        return data



if __name__ == '__main__':
    ''' check I/O shape '''
    print("---Checking I/O shape of the module 'Equalizer'---")
    sr = 44100
    segment_length = sr*10

    eq = Equalizer()
    eq.randomize()

    in_sig = np.random.rand(segment_length, 2)
    out_sig = eq.process(in_sig)
    print(f"input shape : {in_sig.shape}\noutput shape : {out_sig.shape}")
