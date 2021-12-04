from dataclasses import field
import ezmsg as ez

from ezbci.openbci.openbci import OpenBCISource, OpenBCISourceSettings

from .modeltraining import ModelTraining, ModelTrainingSettings
from .preprocessing import Preprocessing, PreprocessingSettings
from .shallowfbcspdecoder import ShallowFBSCPDecoder, ShallowFBCSPDecoderSettings

from typing import Any

class DebugPrint( ez.Unit ):

    INPUT = ez.InputStream( Any )

    @ez.subscriber( INPUT )
    async def on_message( self, message: Any ) -> None:
         print( message )

class HololightSystemSettings( ez.Settings ):
    openbcisource_settings: OpenBCISourceSettings
    decoder_settings: ShallowFBCSPDecoderSettings
    modeltraining_settings: ModelTrainingSettings
    preprocessing_settings: PreprocessingSettings = field(
        default_factory = PreprocessingSettings
    )

class HololightSystem( ez.System ):

    SETTINGS: HololightSystemSettings

    SOURCE = OpenBCISource()
    PREPROC = Preprocessing()
    DECODER = ShallowFBSCPDecoder()
    TRAINING = ModelTraining()

    DEBUG = DebugPrint()

    def configure( self ) -> None:
        self.SOURCE.apply_settings( self.SETTINGS.openbcisource_settings )
        self.PREPROC.apply_settings( self.SETTINGS.preprocessing_settings )
        self.DECODER.apply_settings( self.SETTINGS.decoder_settings )
        self.TRAINING.apply_settings( self.SETTINGS.modeltraining_settings )

    def network( self ) -> ez.NetworkDefinition:
        return ( 
            ( self.SOURCE.OUTPUT_SIGNAL, self.PREPROC.INPUT_SIGNAL ),
            ( self.PREPROC.OUTPUT_SIGNAL, self.DECODER.INPUT_SIGNAL ),
            ( self.TRAINING.OUTPUT_MODEL, self.DECODER.INPUT_MODEL ),
            ( self.DECODER.OUTPUT_DECODE, self.DEBUG.INPUT ),
            ( self.PREPROC.OUTPUT_SIGNAL, self.TRAINING.INPUT_SIGNAL ),
        )


