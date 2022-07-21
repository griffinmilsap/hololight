import asyncio
from dataclasses import field
import logging

import ezmsg.core as ez
import numpy as np
import torch as th

from .shallowfbcspnet import ShallowFBCSPNet
from .sampler import SampleMessage

from typing import AsyncGenerator, List, Optional, Union, Tuple

logger = logging.getLogger( __name__ )

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

from pathlib import Path

from ezmsg.testing.debuglog import DebugLog
from ezmsg.sigproc.window import Window, WindowSettings

from .plotter import EEGPlotter
from .eegsynth import EEGSynth, EEGSynthSettings
from .preprocessing import Preprocessing, PreprocessingSettings
from .training.server import TrainingServer, TrainingServerSettings
from .sampler import Sampler, SamplerSettings

class ShallowFBCSPTrainingTestSystemSettings( ez.Settings ):
    shallowfbcsptraining_settings: ShallowFBCSPTrainingSettings
    trainingserver_settings: TrainingServerSettings
    eeg_settings: EEGSynthSettings = field( 
        default_factory = EEGSynthSettings 
    )
    preproc_settings: PreprocessingSettings = field(
        default_factory = PreprocessingSettings
    )

class ShallowFBCSPTrainingTestSystem( ez.System ):

    SETTINGS: ShallowFBCSPTrainingTestSystemSettings

    EEG = EEGSynth()
    PREPROC = Preprocessing()
    SAMPLER = Sampler()

    WINDOW = Window()
    PLOTTER = EEGPlotter()

    TRAIN_SERVER = TrainingServer()
    FBCSP_TRAINING = ShallowFBCSPTraining()
    DEBUG = DebugLog()

    def configure( self ) -> None:
        self.FBCSP_TRAINING.apply_settings( 
            self.SETTINGS.shallowfbcsptraining_settings 
        )

        self.EEG.apply_settings(
            self.SETTINGS.eeg_settings
        )

        self.PREPROC.apply_settings(
            self.SETTINGS.preproc_settings
        )

        self.TRAIN_SERVER.apply_settings(
            self.SETTINGS.trainingserver_settings
        )

        self.WINDOW.apply_settings(
            WindowSettings( 
                window_dur = 4.0, # sec
                window_shift = 1.0 # sec
            )
        )

        self.SAMPLER.apply_settings(
            SamplerSettings(
                buffer_dur = 5.0 # sec
            )
        )

    def network( self ) -> ez.NetworkDefinition:
        return ( 
            ( self.EEG.OUTPUT_SIGNAL, self.PREPROC.INPUT_SIGNAL ),
            ( self.PREPROC.OUTPUT_SIGNAL, self.SAMPLER.INPUT_SIGNAL ),
            ( self.SAMPLER.OUTPUT_SAMPLE, self.FBCSP_TRAINING.INPUT_SAMPLE ),

            ( self.SAMPLER.OUTPUT_SAMPLE, self.DEBUG.INPUT ),

            ( self.TRAIN_SERVER.OUTPUT_SAMPLETRIGGER, self.SAMPLER.INPUT_TRIGGER ),

            # Plotter connections
            ( self.PREPROC.OUTPUT_SIGNAL, self.WINDOW.INPUT_SIGNAL ),
            ( self.WINDOW.OUTPUT_SIGNAL, self.PLOTTER.INPUT_SIGNAL ),
        )

    def process_components( self ) -> Tuple[ ez.Component, ... ]:
        return ( 
            self.FBCSP_TRAINING, 
            self.PLOTTER, 
            self.TRAIN_SERVER 
        )

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description = 'ShallowFBCSP Dev/Test Environment'
    )

    parser.add_argument(
        '--channels',
        type = int,
        help = "Number of EEG channels to simulate",
        default = 8
    )

    parser.add_argument(
        '--cert',
        type = lambda x: Path( x ),
        help = "Certificate file for frontend server",
        default = ( Path( '.' ) / 'cert.pem' ).absolute()
    )

    parser.add_argument(
        '--key',
        type = lambda x: Path( x ),
        help = "Private key for frontend server [Optional -- assumed to be included in --cert file if omitted)",
        default = None
    )

    parser.add_argument(
        '--cacert',
        type = lambda x: Path( x ),
        help = "Certificate for custom authority [Optional]",
        default = None
    )

    args = parser.parse_args()

    channels: int = args.channels
    cert: Path = args.cert
    key: Optional[ Path ] = args.key
    cacert: Optional[ Path ] = args.cacert

    settings = ShallowFBCSPTrainingTestSystemSettings(
        eeg_settings = EEGSynthSettings(
            fs = 500.0, # Hz
            channels = channels,
            blocksize = 200, # samples per block
            amplitude = 10e-6, # Volts
            dc_offset = 0, # Volts
            alpha = 9.5, # Hz; don't add alpha if None

            # Rate (in Hz) at which to dispatch EEGMessages
            # None => as fast as possible
            # float number => block publish rate in Hz
            # 'realtime' => Use wall-clock to publish EEG at proper rate
            dispatch_rate = 'realtime'
        ),

        trainingserver_settings = TrainingServerSettings(
            cert = cert,
            key = key,
            ca_cert = cacert
        ),

        shallowfbcsptraining_settings = ShallowFBCSPTrainingSettings(
            model_spec = ShallowFBCSPNet(
                in_chans = channels,
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