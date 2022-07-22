from dataclasses import field, replace
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

from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings

from ezmsg.testing.debuglog import DebugLog
from ezmsg.sigproc.window import Window, WindowSettings
from ezmsg.eeg.eegmessage import EEGMessage

from .plotter import EEGPlotter
from .eegsynth import EEGSynth, EEGSynthSettings
from .preprocessing import Preprocessing, PreprocessingSettings
from .training.server import TrainingServer, TrainingServerSettings
from .sampler import Sampler, SamplerSettings

class SampleSignalModulatorSettings( ez.Settings ):
    signal_amplitude: float = 0.01

class SampleSignalModulatorState( ez.State ):
    classes: List[ str ] = field( default_factory = list )

class SampleSignalModulator( ez.Unit ):

    STATE: SampleSignalModulatorState
    SETTINGS: SampleSignalModulatorSettings

    INPUT_SAMPLE = ez.InputStream( SampleMessage )
    OUTPUT_SAMPLE = ez.OutputStream( SampleMessage )

    OUTPUT_EEG = ez.OutputStream( EEGMessage )

    @ez.subscriber( INPUT_SAMPLE )
    @ez.publisher( OUTPUT_SAMPLE )
    @ez.publisher( OUTPUT_EEG )
    async def on_sample( self, msg: SampleMessage ) -> AsyncGenerator:

        if msg.trigger.value not in self.STATE.classes:
            self.STATE.classes.append( msg.trigger.value )

        assert isinstance( msg.sample, EEGMessage )
        sample: EEGMessage = msg.sample

        ch_idx = min( self.STATE.classes.index( msg.trigger.value ), sample.n_ch )
        arr = np.swapaxes( sample.data, sample.time_dim, 0 )

        sample_time = ( np.arange( sample.n_time ) / sample.fs )
        test_signal = np.sin( 2.0 * np.pi * 20.0 * sample_time )
        test_signal = test_signal * np.hamming( sample.n_time )
        arr[:, ch_idx] = arr[:, ch_idx] + ( test_signal * self.SETTINGS.signal_amplitude )

        sample = replace( sample, data = np.swapaxes( arr, sample.time_dim, 0 ) )
        yield self.OUTPUT_EEG, sample
        yield self.OUTPUT_SAMPLE, replace( msg, sample = sample )


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
    INJECTOR = SampleSignalModulator()
    LOGGER = MessageLogger()

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

        self.LOGGER.apply_settings(
            MessageLoggerSettings(
                output = Path( '.' ) / 'recordings' / 'traindata.txt'
            )
        )

    def network( self ) -> ez.NetworkDefinition:
        return ( 
            ( self.EEG.OUTPUT_SIGNAL, self.PREPROC.INPUT_SIGNAL ),
            ( self.PREPROC.OUTPUT_SIGNAL, self.SAMPLER.INPUT_SIGNAL ),
            ( self.SAMPLER.OUTPUT_SAMPLE, self.INJECTOR.INPUT_SAMPLE ),
            ( self.INJECTOR.OUTPUT_SAMPLE, self.FBCSP_TRAINING.INPUT_SAMPLE ),

            ( self.INJECTOR.OUTPUT_SAMPLE, self.DEBUG.INPUT ),

            ( self.TRAIN_SERVER.OUTPUT_SAMPLETRIGGER, self.SAMPLER.INPUT_TRIGGER ),

            # Plotter connections
            ( self.INJECTOR.OUTPUT_EEG, self.PLOTTER.INPUT_SIGNAL ),

            ( self.INJECTOR.OUTPUT_SAMPLE, self.LOGGER.INPUT_MESSAGE ),
        )

    def process_components( self ) -> Tuple[ ez.Component, ... ]:
        return ( 
            self.FBCSP_TRAINING, 
            self.PLOTTER, 
            self.TRAIN_SERVER,
            self.LOGGER
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
            blocksize = 100, # samples per block
            amplitude = 10e-6, # Volts
            dc_offset = 0, # Volts
            alpha = 9.5, # Hz; don't add alpha if None

            # Rate (in Hz) at which to dispatch EEGMessages
            # None => as fast as possible
            # float number => block publish rate in Hz
            # 'realtime' => Use wall-clock to publish EEG at proper rate
            dispatch_rate = 'realtime'
        ),

        preproc_settings = PreprocessingSettings(
            # 1. Bandpass Filter
            bpfilt_order = 5,
            bpfilt_cuton = 5.0, # Hz
            bpfilt_cutoff = 30.0, # Hz

            # 2. Downsample
            downsample_factor = 2, # Downsample factor to reduce sampling rate to ~ 250 Hz

            # 3. Exponentially Weighted Standardization
            ewm_history_dur = 4.0, # sec

            # 4. Sliding Window
            output_window_dur = 1.0, # sec
            output_window_shift = 1.0, # sec
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
                input_time_length = 1000
            )
        )
    )

    system = ShallowFBCSPTrainingTestSystem( settings )

    ez.run_system( system )


# Change TimeseriesMessage to TSMessage
# Change EEGMessage to MultiChTSMessage
# Alias EEGMessage

# Template typing parameter for SampleMessage