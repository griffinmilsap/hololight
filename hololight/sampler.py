from dataclasses import replace
import logging
from multiprocessing.sharedctypes import Value

import ezmsg.core as ez
import numpy as np

from ezmsg.util.stampedmessage import StampedMessage
from ezmsg.sigproc.timeseriesmessage import TimeSeriesMessage

from typing import Optional, Any, Tuple

logger = logging.getLogger( __name__ )

class TriggerInfo( StampedMessage ):
    trigger: Any

class SampleMessage( StampedMessage ):
    trigger_info: TriggerInfo
    sample: TimeSeriesMessage
    time_offset: float

class SamplerSettings( ez.Settings ):
    sample_per: Tuple[ float, float ] # seconds

class SamplerState( ez.State ):
    trigger_info: Optional[ TriggerInfo ] = None # sampling if trigger is not None
    trigger_offset: Optional[ int ] = None
    last_msg: Optional[ TimeSeriesMessage ] = None
    buffer: Optional[ np.ndarray ] = None

class Sampler( ez.Unit ):
    SETTINGS: SamplerSettings
    STATE: SamplerState

    INPUT_TRIGGER = ez.InputStream( Any )
    INPUT_SIGNAL = ez.OutputStream( TimeSeriesMessage )
    OUTPUT_SAMPLE = ez.OutputStream( SampleMessage )

    def initialize( self ) -> None:
        if self.SETTINGS.sample_per[1] <= self.SETTINGS.sample_per[0]:
            raise ValueError( f'Invalid sample period: {self.SETTINGS.sample_per[1]=} <= {self.SETTINGS.sample_per[0]=}' )

    @ez.subscriber( INPUT_TRIGGER )
    async def on_trigger( self, msg: Any ) -> None:
        # TODO: It is probably possible to adapt this code to 
        # allow multiple overlapping sample acquisitions
        if self.STATE.trigger_info is None:
            if self.STATE.last_msg is not None:
                # Do what we can with the wall clock to determine sample alignment
                self.STATE.trigger_info = TriggerInfo( trigger = msg )
                wall_delta = self.STATE.trigger_info._timestamp
                wall_delta -= self.STATE.last_msg._timestamp
                self.STATE.trigger_offset = int( wall_delta * self.STATE.last_msg.fs )
            else: logger.warn( 'Sampling failed: no signal to sample yet' )
        else: logger.warn( 'Sampling failed: already sampling' )

    @ez.subscriber( INPUT_SIGNAL )
    @ez.publisher( OUTPUT_SAMPLE )
    async def on_signal( self, msg: TimeSeriesMessage ) -> None:

        if self.STATE.last_msg is None:
            self.STATE.last_msg = msg

        # Easier to deal with timeseries on axis 0
        last_msg = self.STATE.last_msg
        msg_data = np.swapaxes( msg.data, msg.time_dim, 0 )
        last_msg_data = np.swapaxes( last_msg.data, last_msg.time_dim, 0 )
        
        if ( # Check if signal properties have changed in a breaking way
            msg.fs != last_msg.fs or \
            msg.time_dim != last_msg.time_dim or \
            msg_data.shape[1:] != last_msg_data.shape[1:]
        ):
            # Data stream changed meaningfully -- flush buffer, stop sampling
            if self.STATE.trigger_info is not None:
                logger.warn( 'Sampling failed: signal properties changed' )

            self.STATE.buffer = None
            self.STATE.trigger_info = None
            self.STATE.trigger_offset = None

        # We will report a single sample at minimum
        start_offset = int( self.SETTINGS.sample_per[0] * msg.fs )
        stop_offset = int( self.SETTINGS.sample_per[1] * msg.fs )
        max_buf_len: int = max( ( 0, -start_offset ) ) + 1

        # Accumulate buffer ( time dim => dim 0 )
        self.STATE.buffer = msg_data if self.STATE.buffer is None else \
            np.concatenate( ( self.STATE.buffer, msg_data ), axis = 0 )

        # trigger_offset points to t = 0 within buffer
        if self.STATE.trigger_offset is not None:
            self.STATE.trigger_offset -= msg.n_time

        if self.STATE.trigger_info is not None:
            start = self.STATE.trigger_offset + start_offset
            stop = self.STATE.trigger_offset + stop_offset

            if stop < 0:
                if abs( start ) < self.STATE.buffer.shape[0]:

                    # We should be able to dispatch a sample
                    sample_data = self.STATE.buffer[ start : stop, ... ]
                    sample_data = np.swapaxes( sample_data, msg.time_dim, 0 )
                    
                    yield self.OUTPUT_SAMPLE, SampleMessage( 
                        trigger_info = self.STATE.trigger_info,
                        sample = replace( msg, data = sample_data ),
                        time_offset = start_offset / msg.fs
                    )

                else: logger.warn( 'Sampling failed: insufficient buffer size' )

                self.STATE.trigger_info = None
                self.STATE.trigger_offset = None

        # We only want to prune buffer if we aren't sampling
        elif self.STATE.buffer.shape[ 0 ] > max_buf_len:
            self.STATE.buffer = self.STATE.buffer[ -max_buf_len:, ... ]

        self.STATE.last_msg = msg


## Dev/test apparatus
import asyncio

from ezmsg.testing.debuglog import DebugLog
from ezmsg.sigproc.synth import Oscillator, OscillatorSettings

from typing import AsyncGenerator

class SampleFormatter( ez.Unit ):

    INPUT = ez.InputStream( SampleMessage )
    OUTPUT = ez.OutputStream( str )

    @ez.subscriber( INPUT )
    @ez.publisher( OUTPUT )
    async def format( self, msg: SampleMessage ) -> AsyncGenerator:
        str_msg = f'Trigger: {msg.trigger_info.trigger}, '
        str_msg += f'{msg.sample.n_time} samples @ {msg.sample.fs} Hz, '
        
        time_axis = np.arange(msg.sample.n_time) / msg.sample.fs
        time_axis = time_axis + msg.time_offset
        str_msg += f'[{time_axis[0]},{time_axis[-1]})'
        
        yield self.OUTPUT, str_msg

class TriggerGenerator( ez.Unit ):

    OUTPUT_TRIGGER = ez.OutputStream( int )

    @ez.publisher( OUTPUT_TRIGGER )
    async def generate( self ) -> AsyncGenerator:
        await asyncio.sleep( 0.5 )

        output = 0
        while True:
            yield self.OUTPUT_TRIGGER, output
            await asyncio.sleep( 5.0 )
            output += 1

class SamplerTestSystem( ez.System ):

    SETTINGS: SamplerSettings

    OSC = Oscillator()
    SAMPLER = Sampler()
    TRIGGER = TriggerGenerator()
    FORMATTER = SampleFormatter()
    DEBUG = DebugLog()

    def configure( self ) -> None:
        self.SAMPLER.apply_settings( self.SETTINGS )

        self.OSC.apply_settings( OscillatorSettings(
            n_time = 2, # Number of samples to output per block
            fs = 10,  # Sampling rate of signal output in Hz
            dispatch_rate = 'realtime',
            freq = 2.0,  # Oscillation frequency in Hz
            amp = 1.0,  # Amplitude
            phase = 0.0,  # Phase offset (in radians)
            sync = True, # Adjust `freq` to sync with sampling rate
        ) )

    def network( self ) -> ez.NetworkDefinition:
        return (
            ( self.OSC.OUTPUT_SIGNAL, self.SAMPLER.INPUT_SIGNAL ),
            ( self.TRIGGER.OUTPUT_TRIGGER, self.SAMPLER.INPUT_TRIGGER ),

            ( self.TRIGGER.OUTPUT_TRIGGER, self.DEBUG.INPUT ),
            ( self.SAMPLER.OUTPUT_SAMPLE, self.FORMATTER.INPUT ),
            ( self.FORMATTER.OUTPUT, self.DEBUG.INPUT )
        )

if __name__ == '__main__':

    settings = SamplerSettings( sample_per = ( -10.0, 10.0 ) )
    system = SamplerTestSystem( settings )

    ez.run_system( system )