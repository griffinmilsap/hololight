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

from ezmsg.testing.debuglog import DebugLog
from ezmsg.eeg.eegmessage import EEGMessage
from ezmsg.sigproc.butterworthfilter import ButterworthFilter, ButterworthFilterSettings
from ezmsg.sigproc.window import Window, WindowSettings

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

import matplotlib
matplotlib.use( "agg" )

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

import panel
import param

class EEGPlot( param.Parameterized ):
    # Reactive Parameters
    gain = param.Number( default = 1e6 )
    offset = param.Number( default = 2.0 )
    msg: Optional[ EEGMessage ] = param.Parameter()

    # Internal plotting functionality
    fig: Figure
    ax: Axes
    lines: Optional[ List[ Line2D ] ] = None

    def __init__( self, **params ) -> None:
        super().__init__( **params )
        self.fig, self.ax = plt.subplots( figsize = ( 8.0, 4.0 ) )

    def view( self ) -> Figure:
        if self.msg is not None:
            y_offsets = np.arange( self.msg.n_ch ) * self.offset
            arr = ( self.msg.data * self.gain ) + y_offsets
            eeg_dur = self.msg.n_time / self.msg.fs

            if self.lines is None:
                time = ( np.arange( self.msg.n_time ) / self.msg.fs ) - eeg_dur
                self.lines = self.ax.plot( time, arr )
            else:
                self.ax.set_autoscale_on( False )
                for idx, line in enumerate( self.lines ):
                    line.set_ydata( arr[ :, idx ] )

            self.ax.set_ylim( -self.offset, self.msg.n_ch * self.offset )
            self.ax.set_yticks( y_offsets )
            if self.msg.ch_names is not None:
                self.ax.set_yticklabels( self.msg.ch_names )
            self.ax.set_xlim( -eeg_dur, 0.0 )
            self.ax.grid( 'True' )

            self.ax.spines[ 'right' ].set_visible( False )
            self.ax.spines[ 'top' ].set_visible( False )

            self.fig.canvas.draw()


        return self.fig

    def panel( self ) -> panel.viewable.Viewable:
        return panel.Row(
            self.view,
            panel.Column(
                "<br>\n# EEG Plot",
                panel.widgets.NumberInput.from_param( self.param[ 'gain' ] ),
                panel.widgets.NumberInput.from_param( self.param[ 'offset' ] )
            )
        )

class EEGPlotterSettings( ez.Settings ):
    ...

class EEGPlotterState( ez.State ):
    plot: EEGPlot = field( default_factory = EEGPlot )

class EEGPlotter( ez.Unit ):

    SETTINGS: EEGPlotterSettings
    STATE: EEGPlotterState

    INPUT_SIGNAL = ez.InputStream( EEGMessage )

    @ez.subscriber( INPUT_SIGNAL )
    async def on_signal( self, msg: EEGMessage ) -> None:
        self.STATE.plot.msg = msg

    @ez.main
    def serve_dashboard( self ) -> None:
        panel.serve( self.STATE.plot.panel, port = 8082, show = False )
        

class ShallowFBCSPTrainingTestSystemSettings( ez.Settings ):
    shallowfbcsptraining_settings: ShallowFBCSPTrainingSettings
    eeg_settings: EEGSynthSettings = field( 
        default_factory = EEGSynthSettings 
    )

class ShallowFBCSPTrainingTestSystem( ez.System ):

    SETTINGS: ShallowFBCSPTrainingTestSystemSettings

    EEG = EEGSynth()
    WINDOW = Window()
    PLOTTER = EEGPlotter()
    FBCSP_TRAINING = ShallowFBCSPTraining()
    DEBUG = DebugLog()

    def configure( self ) -> None:
        self.FBCSP_TRAINING.apply_settings( 
            self.SETTINGS.shallowfbcsptraining_settings 
        )

        self.EEG.apply_settings(
            self.SETTINGS.eeg_settings
        )

        self.WINDOW.apply_settings(
            WindowSettings( 
                window_dur = 4.0, 
                window_shift = 0.2 
            )
        )

    def network( self ) -> ez.NetworkDefinition:
        return ( 
            ( self.EEG.OUTPUT_SIGNAL, self.WINDOW.INPUT_SIGNAL ),
            # ( self.WINDOW.OUTPUT_SIGNAL, self.DEBUG.INPUT ),
            ( self.WINDOW.OUTPUT_SIGNAL, self.PLOTTER.INPUT_SIGNAL ),
        )

    def process_components( self ) -> Tuple[ ez.Component, ... ]:
        return ( self.FBCSP_TRAINING, self.PLOTTER, )

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
# Dashboard with Panel for timeseries visualization
# Adapt websocket API to send 
    # LOG
    # LOGJSON
    # TRIGGER
        # (Incl Per and value)
# Send two triggers from individual labjs slides