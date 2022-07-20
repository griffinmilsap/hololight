import asyncio
from dataclasses import field
from distutils.log import Log

import ezmsg.core as ez
import numpy as np
import torch as th

from .shallowfbcspnet import ShallowFBCSPNet
from .sampler import SampleMessage

from typing import AsyncGenerator, List, Optional, Union, Tuple

class ShallowFBCSPTrainingSettings( ez.Settings ):
    model_spec: ShallowFBCSPNet

class ShallowFBCSPTrainingState( ez.State ):
    samples: List[ SampleMessage ] = field( default_factory = list )

class ShallowFBCSPTraining( ez.Unit ):
    SETTINGS: ShallowFBCSPTrainingSettings
    STATE: ShallowFBCSPTrainingState

    INPUT_SAMPLE = ez.InputStream( SampleMessage )

    @ez.subscriber( INPUT_SAMPLE )
    async def on_sample( self, msg: SampleMessage ) -> None:
        self.STATE.samples.append( msg )


# Dev/Test Fixture

from ezmsg.testing.debuglog import DebugLog
from ezmsg.eeg.eegmessage import EEGMessage
from ezmsg.sigproc.butterworthfilter import ButterworthFilter, ButterworthFilterSettings

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
                arr = ( arr.T + ( alpha * self.SETTINGS.amplitude ) ).T

            self.STATE.cur_sample += self.SETTINGS.blocksize

            yield self.OUTPUT_SIGNAL, EEGMessage( 
                data = arr, 
                fs = self.SETTINGS.fs, 
                ch_names = ch_names 
            )

            if self.SETTINGS.dispatch_rate:
                if self.SETTINGS.dispatch_rate == 'realtime':
                    await asyncio.sleep( self.SETTINGS.blocksize / self.SETTINGS.fs )
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

class ShallowFBCSPTrainingTestSystemSettings( ez.Settings ):
    shallowfbcsptraining_settings: ShallowFBCSPTrainingSettings
    eeg_settings: EEGSynthSettings = field( 
        default_factory = EEGSynthSettings 
    )

class ShallowFBCSPTrainingTestSystem( ez.System ):

    SETTINGS: ShallowFBCSPTrainingTestSystemSettings

    EEG = EEGSynth()
    FBCSP_TRAINING = ShallowFBCSPTraining()
    DEBUG = DebugLog()

    def configure( self ) -> None:
        self.FBCSP_TRAINING.apply_settings( 
            self.SETTINGS.shallowfbcsptraining_settings 
        )

    def network( self ) -> ez.NetworkDefinition:
        return ( 
            ( self.EEG.OUTPUT_SIGNAL, self.DEBUG.INPUT ), 
        )

    def process_components( self ) -> Tuple[ ez.Component, ... ]:
        return ( self.FBCSP_TRAINING, )

if __name__ == '__main__':

    num_channels = 8

    settings = ShallowFBCSPTrainingTestSystemSettings(
        eeg_settings = EEGSynthSettings(
            fs = 500.0, # Hz
            channels = num_channels,
            blocksize = 50, # samples per block
            amplitude = 10e-6, # Volts
            dc_offset = 0, # Volts
            alpha = 9.5, # Hz; don't add alpha if None

            # Rate (in Hz) at which to dispatch EEGMessages
            # None => as fast as possible
            # float number => block publish rate in Hz
            # 'realtime' => Use wall-clock to publish EEG at proper rate
            dispatch_rate = 'realtime'
        ),

        shallowfbcsptraining_settings = ShallowFBCSPTrainingSettings(
            model_spec = ShallowFBCSPNet(
                in_chans = num_channels,
                n_classes = 2,
            )
        )
    )

    system = ShallowFBCSPTrainingTestSystem( settings )

    ez.run_system( system )


# Change TimeseriesMessage to TSMessage
# Change EEGMessage to MultiChTSMessage
# Alias EEGMessage
# Adapt ShallowFBCSPNet to accept a n_crops parameter INSTEAD of a final_conv_length parameter
# Split ShallowFBCSPNet dataclass into two sets of parameters:
    # one for model specifications
    # one for IO parameters (num channels, num crops, input time len, etc)
# Figure out how to elevate HTTPS Connection to websocket on same port
# Dashboard with Panel for timeseries visualization
# Adapt websocket API to send 
    # LOG
    # LOGJSON
    # TRIGGER
        # (Incl Per and value)
# Send two triggers from individual labjs slides