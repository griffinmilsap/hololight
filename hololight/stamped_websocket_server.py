import ezmsg.core as ez
from ezmsg.websocket import WebsocketServer, WebsocketSettings
from ezmsg.util.stampedmessage import StampedMessage
from typing import ByteString, AsyncGenerator

class StampedTextMessage(StampedMessage):
    text: str

class Stamper( ez.Unit ):
    INPUT = ez.InputStream(ByteString)
    OUTPUT = ez.OutputStream(StampedTextMessage)

    @ez.subscriber(INPUT)
    @ez.publisher(OUTPUT)
    async def stamp_message( self, message: ByteString ) -> AsyncGenerator:
        yield ( self.OUTPUT, StampedTextMessage( text = message ))

class StampedWebsocketServer( ez.Collection ):

    SETTINGS: WebsocketSettings

    OUTPUT_MESSAGE = ez.OutputStream(StampedTextMessage)

    WS_SERVER = WebsocketServer()
    STAMPER = Stamper()

    def configure( self ) -> None:
        self.WS_SERVER.apply_settings(self.SETTINGS)

    def network( self ) -> ez.NetworkDefinition:
        return (
            ( self.WS_SERVER.OUTPUT, self.STAMPER.INPUT ),
            ( self.STAMPER.OUTPUT, self.OUTPUT_MESSAGE )
        )
