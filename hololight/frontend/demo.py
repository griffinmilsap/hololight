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

from typing import AsyncGenerator, Optional, List

logger = logging.getLogger( __name__ )


class HololightDemoSettings( ez.Settings ):
    cert: Path
    bridge_host: str
    host: str = '0.0.0.0'
    port: int = 443
    ws_port: int = 8082
    trigger_class: int = 1
    trigger_thresh: float = 0.9


class HololightDemoState( ez.State ):
    focus_light: Optional[ str ] = None
    decode_class: Optional[ int ] = None
    bridge: Optional[ Bridge ] = None


class HololightDemo( ez.Unit ):

    SETTINGS: HololightDemoSettings
    STATE: HololightDemoState

    INPUT_DECODE = ez.InputStream( DecoderOutput )

    def initialize( self ) -> None:
        bridge = Bridge( self.SETTINGS.bridge_host )
        bridge.connect()
        self.STATE.bridge = bridge
        lights: List[ Light ] = self.STATE.bridge.lights
        for light in lights:
            light.on = False

    @ez.task
    async def start_websocket_server( self ) -> None:

        async def connection( websocket: websockets.server.WebSocketServerProtocol, path ):
            logger.info( 'Client Connected to Websocket Input' )

            async def blink_light( light: Light ) -> None:
                while True:
                    await asyncio.sleep( 1.0 )
                    light.on = not light.on

            try:
                while True:
                    data = await websocket.recv()

                    cmd, value = data.split( ': ' )

                    if cmd == 'COMMAND':

                        if value == 'START_MAPPING': # Perform Spatial Light Mapping
                            logger.info( 'Starting Light Mapping Sequence...' )
                            lights: List[ Light ] = self.STATE.bridge.lights
                            for light in lights:
                                if not light.reachable: continue

                                logger.info( f'Asking client to locate {light.name}' )
                                await websocket.send( f'LOCATE: {light.name}' )

                                blink_task = asyncio.create_task( blink_light( light ) )
                                data = await websocket.recv() # Echo Light Name
                                logger.info( f'Client responds {data}' )
                                blink_task.cancel()
                            
                            logger.info( 'Light Mapping Sequence Complete' )
                            await websocket.send( 'RESULT: DONE_MAPPING' )

                    elif cmd == 'SELECT': # Select/Focus a light
                        self.STATE.focus_light = value if value != 'null' else None
                        logger.info( f'Client focusing light {self.STATE.focus_light}' )

                    else:
                        logger.info( 'Received problematic message from websocket client: {data}')

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


    @ez.subscriber( INPUT_DECODE )
    async def on_decode( self, decode: DecoderOutput ) -> None:

        probs = np.exp( decode.output )
        cur_class = probs.argmax()
        cur_prob = probs[ cur_class ]

        logger.info( f'Decoder: {cur_class} @ {cur_prob}' )

        if self.STATE.decode_class is None:
            self.STATE.decode_class = cur_class

        if cur_class != self.STATE.decode_class:
            if cur_prob > self.SETTINGS.trigger_thresh:
                self.STATE.decode_class = cur_class
                if cur_class == self.SETTINGS.trigger_class:
                    if self.STATE.focus_light is not None:
                        try:
                            light: Light = self.STATE.bridge.lights_by_name[ self.STATE.focus_light ]
                            light.on = not light.on
                        except KeyError:
                            logger.warn( f'Light {self.STATE.focus_light} does not exist.' )

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

class GenerateDecodeOutput( ez.Unit ):

    OUTPUT_DECODE = ez.OutputStream( DecoderOutput )

    @ez.publisher( OUTPUT_DECODE )
    async def generate( self ) -> AsyncGenerator:
        output = np.array( [ True, False ] )
        while True:
            out = ( output.astype( float ) * 0.9 ) + 0.05
            out = np.log( out / out.sum() )
            yield self.OUTPUT_DECODE, DecoderOutput( out )
            await asyncio.sleep( 2.0 )
            output = ~output

class HololightTestSystem( ez.System ):

    SETTINGS: HololightDemoSettings

    HOLOLIGHT = HololightDemo()
    DECODE_TEST = GenerateDecodeOutput()

    def configure( self ) -> None:
        return self.HOLOLIGHT.apply_settings( self.SETTINGS )

    def network( self ) -> ez.NetworkDefinition:
        return (
            ( self.DECODE_TEST.OUTPUT_DECODE, self.HOLOLIGHT.INPUT_DECODE ),
        )

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description = 'Hololight Test Script'
    )

    parser.add_argument(
        '--bridge',
        type = str,
        help = 'Hostname for Philips Hue Bridge'
    )

    parser.add_argument(
        '--cert',
        type = lambda x: Path( x ),
        help = "Certificate file for frontend server",
        default = ( Path( '.' ) / 'cert.pem' ).absolute()
    )

    args = parser.parse_args()

    bridge_host: str = args.bridge
    cert: Path = args.cert

    settings = HololightDemoSettings(
        cert = cert,
        bridge_host = bridge_host
    )

    system = HololightTestSystem( settings )
    ez.run_system( system )



