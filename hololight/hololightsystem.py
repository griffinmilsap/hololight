from dataclasses import field
import ezmsg.core as ez

from ezmsg.eeg.openbci import OpenBCISource, OpenBCISourceSettings

from .modeltraining import ModelTraining, ModelTrainingSettings
from .preprocessing import Preprocessing, PreprocessingSettings
from .shallowfbcspdecoder import ShallowFBSCPDecoder, ShallowFBCSPDecoderSettings
from .hue import HueDemo, HueDemoSettings

from typing import Any, Tuple

class DebugPrint( ez.Unit ):

    INPUT = ez.InputStream( Any )

    @ez.subscriber( INPUT )
    async def on_message( self, message: Any ) -> None:
         print( message )

class HololightSystemSettings( ez.Settings ):
    openbcisource_settings: OpenBCISourceSettings
    decoder_settings: ShallowFBCSPDecoderSettings
    modeltraining_settings: ModelTrainingSettings
    huedemo_settings: HueDemoSettings = field(
        default_factory = HueDemoSettings
    )
    preprocessing_settings: PreprocessingSettings = field(
        default_factory = PreprocessingSettings
    )

class HololightSystem( ez.System ):

    SETTINGS: HololightSystemSettings

    SOURCE = OpenBCISource()
    PREPROC = Preprocessing()
    DECODER = ShallowFBSCPDecoder()
    TRAINING = ModelTraining()
    HUE = HueDemo()

    DEBUG = DebugPrint()

    def configure( self ) -> None:
        self.SOURCE.apply_settings( self.SETTINGS.openbcisource_settings )
        self.PREPROC.apply_settings( self.SETTINGS.preprocessing_settings )
        self.DECODER.apply_settings( self.SETTINGS.decoder_settings )
        self.TRAINING.apply_settings( self.SETTINGS.modeltraining_settings )
        self.HUE.apply_settings( self.SETTINGS.huedemo_settings )

    def network( self ) -> ez.NetworkDefinition:
        return ( 
            ( self.SOURCE.OUTPUT_SIGNAL, self.PREPROC.INPUT_SIGNAL ),
            ( self.PREPROC.OUTPUT_SIGNAL, self.DECODER.INPUT_SIGNAL ),
            # ( self.PREPROC.OUTPUT_SIGNAL, self.DEBUG.INPUT ),
            ( self.TRAINING.OUTPUT_MODEL, self.DECODER.INPUT_MODEL ),
            # ( self.DECODER.OUTPUT_DECODE, self.DEBUG.INPUT ),
            ( self.DECODER.OUTPUT_DECODE, self.HUE.INPUT_DECODE ),
            ( self.PREPROC.OUTPUT_SIGNAL, self.TRAINING.INPUT_SIGNAL ),
        )

    def process_components( self ) -> Tuple[ ez.Component, ... ]:
        return ( self.HUE, self.TRAINING )


