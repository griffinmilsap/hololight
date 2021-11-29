from dataclasses import field
import ezmsg as ez

from ezbci.openbci.openbci import OpenBCISource, OpenBCISourceSettings

from .preprocessing import Preprocessing, PreprocessingSettings

from typing import Any

class DebugPrint( ez.Unit ):

    INPUT = ez.InputStream( Any )

    @ez.subscriber( INPUT )
    async def on_message( self, message: Any ) -> None:
         print( message )

class HololightSystemSettings( ez.Settings ):
    openbcisource_settings: OpenBCISourceSettings
    preprocessing_settings: PreprocessingSettings = field(
        default_factory = PreprocessingSettings
    )

class HololightSystem( ez.System ):

    SETTINGS: HololightSystemSettings

    SOURCE = OpenBCISource()
    PREPROC = Preprocessing()
    DEBUG = DebugPrint()

    def configure( self ) -> None:
        self.SOURCE.apply_settings( self.SETTINGS.openbcisource_settings )
        self.PREPROC.apply_settings( self.SETTINGS.preprocessing_settings )

    def network( self ) -> ez.NetworkDefinition:
        return ( 
            ( self.SOURCE.OUTPUT_SIGNAL, self.PREPROC.INPUT_SIGNAL ),
            ( self.PREPROC.OUTPUT_SIGNAL, self.DEBUG.INPUT )
        )


