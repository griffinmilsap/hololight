from dataclasses import field

from json.encoder import JSONEncoder
import json
import base64
from io import TextIOWrapper

from pathlib import Path

import ezmsg as ez
import numpy as np

from ezbci.timeseriesmessage import TimeSeriesMessage, TimeSeriesInfoMessage

from typing import (
    Optional,
    Dict,
    Any
)

class MessageEncoder( json.JSONEncoder ):
    def default( self, obj ):
        if isinstance( obj, ez.Message ):
            return dict( _type = obj.__class__.__name__, **obj.__dict__ )
        elif isinstance( obj, np.ndarray ):
            return [ str( obj.dtype ), base64.b64encode( obj ).decode( 'ascii' ), obj.shape ]
        return JSONEncoder.default( self, obj )

class MessageLoggerSettings( ez.Settings ):
    output: Optional[ Path ] = None

class MessageLoggerState( ez.State ):
    signal_info: Optional[ TimeSeriesInfoMessage ] = None
    output_files: Dict[ Path, TextIOWrapper ] = field( default_factory = dict )

class MessageLogger( ez.Unit ):
    # FIXME: This logger became coupled to timeseriesmessage the moment 
    # I converted it to dynamically log to files.
    # Timeseries messages have metainformation that is only sent once at the
    # beginning of the stream.  This information is required for interpretation
    # of subsequent TimeSeriesDataMessages...  As such, we need to cache the
    # info messages here and write them to files as we dynamically open them.
    # This is terrible for a variety of reasons.
    # Additionally, if we try to write several Timeseries messages to the same file,
    # we don't have any way to distinguish the signals, let alone write both info messages..

    SETTINGS: MessageLoggerSettings
    STATE: MessageLoggerState

    INPUT_START = ez.InputStream( Path )
    INPUT_STOP = ez.InputStream( Path )
    INPUT_SIGNAL = ez.InputStream( TimeSeriesMessage ) # supports logging ONE stream
    INPUT_MESSAGE = ez.InputStream( Any )
    OUTPUT_START = ez.OutputStream( Path )
    OUTPUT_STOP = ez.OutputStream( Path )

    def open_file( self, filepath: Path ) -> Optional[ Path ]:
        """ Returns file path if file successfully opened, otherwise None """
        if filepath in self.STATE.output_files:
            # If the file is already open, we return None
            return None

        if not filepath.parent.exists():
            filepath.parent.mkdir( parents = True )
        self.STATE.output_files[ filepath ] = open( filepath, mode = 'w' )

        # We need to write cacheed signal info if we have it.. ugh this sucks
        if self.STATE.signal_info is not None:
            self.write_message( self.STATE.signal_info )

        return filepath

    def close_file( self, filepath: Path ) -> Optional[ Path ]:
        """ Returns file path if file successfully closed, otherwise None """
        if filepath not in self.STATE.output_files:
            # We haven't opened this file
            return None
        
        self.STATE.output_files[ filepath ].close()
        del self.STATE.output_files[ filepath ]

        return filepath

    def initialize( self ) -> None:
        """ Note that files defined at startup are not published to outputs"""
        if self.SETTINGS.output is not None:
            self.open_file( self.SETTINGS.output )

    @ez.subscriber( INPUT_START )
    @ez.publisher( OUTPUT_START )
    async def start_file( self, message: Path ):
        out = self.open_file( message )
        if out is not None:
            yield ( self.OUTPUT_START, out )

    @ez.subscriber( INPUT_STOP )
    @ez.publisher( OUTPUT_STOP )
    async def stop_file( self, message: Path ):
        out = self.close_file( message )
        if out is not None:
            yield ( self.OUTPUT_STOP, out )

    @ez.subscriber( INPUT_MESSAGE )
    async def on_message( self, message: Any ) -> None:
        self.write_message( message )

    @ez.subscriber( INPUT_SIGNAL )
    async def on_signal( self, message: TimeSeriesMessage ) -> None:
        # We need to write cache signal info.. ugh this sucks
        if isinstance( message, TimeSeriesInfoMessage ):
            self.STATE.signal_info = message
        self.write_message( message )

    def write_message( self, message: Any ) -> None:
        message: str = json.dumps( message, cls = MessageEncoder )
        for output_f in self.STATE.output_files.values():
            output_f.write( f'{message}\n' )
            output_f.flush()


    def shutdown( self ) -> None:
        """ Note that files that are closed at shutdown don't publish messages """
        for filepath in list( self.STATE.output_files ):
            self.close_file( filepath )

    @ez.main
    def dummy( self ) -> None:
        # We could be file I/O bound here
        # So we force a separate process
        pass