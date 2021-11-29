from dataclasses import field
import ezmsg as ez

from ezbci.openbci.openbci import OpenBCISource, OpenBCISourceSettings

from typing import Any

class DebugPrint( ez.Unit ):

    INPUT = ez.InputStream( Any )

    @ez.subscriber( INPUT )
    async def on_message( self, message: Any ) -> None:
         print( message )

class HololightSystemSettings( ez.Settings ):
    openbcisource_settings: OpenBCISourceSettings

class HololightSystem( ez.System ):

    SETTINGS: HololightSystemSettings

    SOURCE = OpenBCISource()
    DEBUG = DebugPrint()

    def configure( self ) -> None:
        self.SOURCE.apply_settings( self.SETTINGS.openbcisource_settings )

    def network( self ) -> ez.NetworkDefinition:
        return ( 
            ( self.SOURCE.OUTPUT_SIGNAL, self.DEBUG.INPUT ),
        )


