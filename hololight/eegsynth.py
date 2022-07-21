import asyncio

import numpy as np
import ezmsg.core as ez

from ezmsg.eeg.eegmessage import EEGMessage
from ezmsg.sigproc.butterworthfilter import ButterworthFilter, ButterworthFilterSettings

from typing import Optional, Union, AsyncGenerator

class EEGSynthSettings( ez.Settings ):
    fs: float = 500.0 # Hz
    channels: int = 8
    blocksize: int = 50 # samples per block
    amplitude: float = 10e-6 # Volts
    dc_offset: float = 0 # Volts

    alpha: Optional[ float ] = 9.5 # Hz; don't add alpha if None

    # Rate (in Hz) at which to dispatch EEGMessages
    # None => as fast as possible
    # float number => block publish rate in Hz
    # 'realtime' => Use wall-clock to publish EEG at proper rate
    dispatch_rate: Optional[ Union[ float, str ] ] = None

class WhiteEEGState( ez.State ):
    cur_sample: int = 0

class WhiteEEG( ez.Unit ):
    SETTINGS: EEGSynthSettings
    STATE: WhiteEEGState

    OUTPUT_SIGNAL = ez.OutputStream( EEGMessage )

    @ez.publisher( OUTPUT_SIGNAL )
    async def generate( self ) -> AsyncGenerator:

        ch_names = [ f'ch_{i+1}' for i in range( self.SETTINGS.channels ) ]
        shape = ( self.SETTINGS.blocksize, self.SETTINGS.channels )
        time = np.arange( self.SETTINGS.blocksize )

        while True:
            arr = np.random.normal( 
                loc = self.SETTINGS.dc_offset, 
                scale = self.SETTINGS.amplitude, 
                size = shape 
            )

            if self.SETTINGS.alpha: # If we're adding fake EEG alpha
                cur_time = ( time + self.STATE.cur_sample ) / self.SETTINGS.fs
                alpha = np.sin( 2.0 * np.pi * self.SETTINGS.alpha * cur_time )
                arr = ( arr.T + ( alpha * self.SETTINGS.amplitude * 0.2 ) ).T

            self.STATE.cur_sample += self.SETTINGS.blocksize

            yield self.OUTPUT_SIGNAL, EEGMessage( 
                data = arr, 
                fs = self.SETTINGS.fs, 
                ch_names = ch_names 
            )

            if self.SETTINGS.dispatch_rate:
                if self.SETTINGS.dispatch_rate == 'realtime':
                    sleep_sec = self.SETTINGS.blocksize / self.SETTINGS.fs
                    await asyncio.sleep( sleep_sec )
                else:
                    await asyncio.sleep( self.SETTINGS.dispatch_rate )

class EEGSynth( ez.Collection ):

    SETTINGS: EEGSynthSettings

    OUTPUT_SIGNAL = ez.OutputStream( EEGMessage )

    WHITE_EEG = WhiteEEG()

    # Filter gives "WhiteEEG" a 1/f "Pink" spectral distribution
    FILTER = ButterworthFilter(
        ButterworthFilterSettings(
            order = 1,
            cutoff = 0.5, # Hz
        )
    )

    def configure( self ) -> None:
        self.WHITE_EEG.apply_settings( self.SETTINGS )

    def network( self ) -> ez.NetworkDefinition:
        return (
            ( self.WHITE_EEG.OUTPUT_SIGNAL, self.FILTER.INPUT_SIGNAL ),
            ( self.FILTER.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL )
        )
     