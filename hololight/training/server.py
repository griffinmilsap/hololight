import asyncio
import http.server
import ssl
import logging

from phue import Bridge, Light

from pathlib import Path

import ezmsg.core as ez
import numpy as np

import websockets
import websockets.server
import websockets.exceptions

from ..shallowfbcspdecoder import DecoderOutput
from ..sampler import SampleTriggerMessage

from typing import AsyncGenerator, Optional, List

logger = logging.getLogger( __name__ )

class TrainingServerSettings( ez.Settings ):
    cert: Path
    host: str = '0.0.0.0'
    port: int = 8080
    ws_port: int = 5545

class TrainingServer( ez.Unit ):

    SETTINGS: TrainingServerSettings

    OUTPUT_SAMPLETRIGGER = ez.InputStream( SampleTriggerMessage )

    @ez.task
    async def start_websocket_server( self ) -> None:

        async def connection( websocket: websockets.server.WebSocketServerProtocol, path ):
            logger.info( 'Client Connected to Websocket Input' )

            try:
                while True:
                    data = await websocket.recv()
                    logger.info( data )

            except websockets.exceptions.ConnectionClosedOK:
                logger.info( 'Websocket Client Closed Connection' )
            except asyncio.CancelledError:
                logger.info( 'Websocket Client Handler Task Cancelled!' )
            except Exception as e:
                logger.warn( 'Error in websocket server:', e )
            finally:
                logger.info( 'Websocket Client Handler Task Concluded' )

        try:
            ssl_context = ssl.SSLContext( ssl.PROTOCOL_TLS_SERVER ) 
            ssl_context.load_cert_chain( 
                certfile = self.SETTINGS.cert, 
                keyfile = self.SETTINGS.cert 
            )

            server = await websockets.server.serve(
                connection,
                self.SETTINGS.host,
                self.SETTINGS.ws_port,
                ssl = ssl_context
            )

            await server.wait_closed()

        finally:
            logger.info( 'Closing Websocket Server' )

    @ez.main
    def serve( self ):

        directory = str( ( Path( __file__ ).parent / 'web' ) )

        class Handler( http.server.SimpleHTTPRequestHandler ):
            def __init__( self, *args, **kwargs ):
                super().__init__( *args, directory = directory, **kwargs )

        address = ( self.SETTINGS.host, self.SETTINGS.port )
        httpd = http.server.HTTPServer( address, Handler )

        httpd.socket = ssl.wrap_socket(
            httpd.socket,
            server_side = True,
            certfile = self.SETTINGS.cert,
            ssl_version = ssl.PROTOCOL_TLS_SERVER
        )

        httpd.serve_forever()


### DEV/TEST APPARATUS

from ezmsg.testing.debuglog import DebugLog

class TrainingServerTestSystem( ez.System ):

    SETTINGS: TrainingServerSettings

    SERVER = TrainingServer()
    DEBUG = DebugLog()

    def configure( self ) -> None:
        return self.SERVER.apply_settings( self.SETTINGS )

    def network( self ) -> ez.NetworkDefinition:
        return (
            ( self.SERVER.OUTPUT_SAMPLETRIGGER, self.DEBUG.INPUT ),
        )

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description = 'Training Server Test Script'
    )

    parser.add_argument(
        '--cert',
        type = lambda x: Path( x ),
        help = "Certificate file for frontend server",
        default = ( Path( '.' ) / 'cert.pem' ).absolute()
    )

    args = parser.parse_args()

    cert: Path = args.cert

    settings = TrainingServerSettings(
        cert = cert,
    )

    system = TrainingServerTestSystem( settings )
    ez.run_system( system )



