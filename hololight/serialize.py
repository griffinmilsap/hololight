import ezmsg as ez

import pickle

from typing import (
    Any,
    AsyncGenerator
)

class Serialize( ez.Unit ):

    INPUT = ez.InputStream( Any )
    OUTPUT_BYTES = ez.OutputStream( bytes )

    @ez.subscriber( INPUT )
    @ez.publisher( OUTPUT_BYTES )
    async def serialize( self, message: Any ) -> AsyncGenerator:
        yield ( self.OUTPUT_BYTES, pickle.dumps( message, protocol = -1 ) )

class Deserialize( ez.Unit ):

    INPUT_BYTES = ez.InputStream( bytes )
    OUTPUT = ez.OutputStream( Any )

    @ez.subscriber( INPUT_BYTES )
    @ez.publisher( OUTPUT )
    async def deserialize( self, message: bytes ) -> AsyncGenerator:
        yield( self.OUTPUT, pickle.loads( message ) )
