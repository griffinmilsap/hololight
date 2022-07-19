import asyncio
import http.server
import ssl
import logging
import json

from pathlib import Path
from dataclasses import field

import ezmsg.core as ez

import websockets
import websockets.server
import websockets.exceptions

from ..sampler import SampleTriggerMessage

from typing import AsyncGenerator, Optional, List

logger = logging.getLogger( __name__ )

class TrainingServerSettings( ez.Settings ):
    cert: Path
    key: Optional[ Path ] = None
    ca_cert: Optional[ Path ] = None
    host: str = '0.0.0.0'
    port: int = 8080
    ws_port: int = 5545
    trigger_sender: Optional[ str ] = None # If specified we only trigger when sender == this string
    assign_trigger_value: bool = False # If true, we assign the commit to the value of the trigger

class TrainingServerState( ez.State ):
    trigger_queue: "asyncio.Queue[ SampleTriggerMessage ]" = field( default_factory = asyncio.Queue )

class TrainingServer( ez.Unit ):

    SETTINGS: TrainingServerSettings
    STATE: TrainingServerState

    OUTPUT_SAMPLETRIGGER = ez.InputStream( SampleTriggerMessage )

    @ez.publisher( OUTPUT_SAMPLETRIGGER ) 
    async def publish_trigger( self ) -> AsyncGenerator:
        while True:
            output = await self.STATE.trigger_queue.get()
            yield self.OUTPUT_SAMPLETRIGGER, output

    @ez.task
    async def start_websocket_server( self ) -> None:

        async def connection( websocket: websockets.server.WebSocketServerProtocol, path ):
            logger.info( 'Client Connected to Websocket Input' )

            try:
                while True:
                    data = json.loads( await websocket.recv() )

                    if self.SETTINGS.trigger_sender is None or ( 
                        self.SETTINGS.trigger_sender is not None and \
                        data[ 'sender' ] == self.SETTINGS.trigger_sender 
                    ):             
                        self.STATE.trigger_queue.put_nowait(
                            SampleTriggerMessage(
                                value = data if self.SETTINGS.assign_trigger_value else None
                            )
                        )

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

            if self.SETTINGS.ca_cert:
                ssl_context.load_verify_locations( self.SETTINGS.ca_cert )

            ssl_context.load_cert_chain( 
                certfile = self.SETTINGS.cert, 
                keyfile = self.SETTINGS.key 
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
            keyfile = self.SETTINGS.key,
            ca_certs = self.SETTINGS.ca_cert,
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

    cert: Path = args.cert
    key: Optional[ Path ] = args.key
    cacert: Optional[ Path ] = args.cacert

    settings = TrainingServerSettings(
        cert = cert,
        key = key,
        ca_cert = cacert,
        trigger_sender = 'Fixation'
    )

    system = TrainingServerTestSystem( settings )
    ez.run_system( system )



