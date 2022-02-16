"""
    Implementation of 'Mastering Effects Manipulator'
        from "End-to-end Music Remastering System Using Self-supervised And Adversarial Training"
        refer to the work at https://dg22302.github.io/MusicRemasteringSystem/
    
    The processing procedure of the Mastering Effects Manipulator module follows "pymixconsole"
        the collection of audio effects modules presented in "Automatic multitrack mixing with a differentiable mixing console of neural audio effects"
        refer to the paper at https://arxiv.org/abs/2010.10291
    refer to pymixconsole at https://github.com/csteinmetz1/pymixconsole
"""
from pymixconsole.processor_list import ProcessorList

import numpy as np
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from mastering_manipulator.gain import Gain
from mastering_manipulator.equalizer import Equalizer
from mastering_manipulator.imager import CrossoverMidSideImager
from mastering_manipulator.limiter import LimiterTypeA



# Mastering Effects Manipulator module 
class Mastering_Effects_Manipulator():
    def __init__(self, block_size=2**17):
        self.block_size = block_size
        self.sample_rate = 44100

        ''' processors for Mastering Effects Manipulator '''
        # PRE : fixed input gain at -8.0 dB
        self.processors_pre = ProcessorList(block_size=self.block_size, sample_rate=self.sample_rate)
        self.processors_pre.add(Gain(gain=-8.0, block_size=self.block_size, sample_rate=self.sample_rate))

        # CORE(random) : EQ -> Imager -> Limiter
        self.processors_core = ProcessorList(block_size=self.block_size, sample_rate=self.sample_rate)
        self.processors_core.add(Equalizer(block_size=self.block_size, sample_rate=self.sample_rate, gain_range=(-15.0, 10.0)))
        self.processors_core.add(CrossoverMidSideImager(crossover_order=4, block_size=self.block_size, sample_rate=self.sample_rate))
        self.processors_core.add(LimiterTypeA(block_size=self.block_size, sample_rate=self.sample_rate))

        self.pre_processors = self.processors_pre.get_all()
        self.core_processors = self.processors_core.get_all()


    # manipulate all audio signals in the input buffer list using the same randomized parameters
    def process(self, buffer_list, randomize = None, reset = None):
        main_processors = self.pre_processors + self.core_processors

        if randomize :
            for processor in self.core_processors:
                processor.randomize()
        else :
            pass
 
        output_buff_list = []
        for buffer in buffer_list:
            for processor in main_processors : 
                buffer = processor.process(buffer)
            output_buff_list.append(buffer)

        if reset :
            for processor in main_processors:
                processor.reset()

        return output_buff_list



if __name__ == "__main__":
    ''' check I/O shape '''
    print("---Checking I/O shape of the module 'Mastering Effects Manipulator'---")
    sr = 44100
    segment_length = sr*10

    manipulator = Mastering_Effects_Manipulator()

    segment_a = np.random.rand(segment_length, 2)
    segment_b = np.random.rand(segment_length, 2)

    man_a = manipulator.process([segment_a], randomize=True, reset=True)[0]
    print(f"Input shape : {segment_a.shape}\nOutput shape : {man_a.shape}")

    # check if the randomized parameter is correctly applied
    output_buff_list = manipulator.process([segment_a, segment_b, segment_a], randomize=True, reset=True)
    print(np.array_equal(output_buff_list[0], man_a))
    print(np.array_equal(output_buff_list[0], output_buff_list[2]))
