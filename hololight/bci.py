import time
from pathlib import Path

import ezmsg as ez

from ezbci.openbci.openbci import (
    OpenBCISourceSettings, 
    GainState,
    PowerStatus,
    BiasSetting,
    OpenBCIChannelConfigSettings,
    OpenBCIChannelSetting,
)
from ezmsg.builtins.websocket import WebsocketSettings

from hololight.go_task import GoTaskSettings
from hololight.modeltraining import ModelTrainingLogicSettings, ModelTrainingSettings, TestSignalInjectorSettings

from .messagelogger import MessageLoggerSettings
from .shallowfbcspdecoder import ShallowFBCSPDecoderSettings
from .hololightsystem import HololightSystem, HololightSystemSettings

from typing import (
    Dict,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description = 'Hololight ARBCI Lightbulb Demonstration'
    )

    ## OpenBCI Arguments
    parser.add_argument(
        '--device',
        type = str,
        help = 'Serial port to pull data from',
        default = 'simulator'
    )

    parser.add_argument(
        '--blocksize',
        type = int,
        help = 'Sample block size @ 500 Hz',
        default = 100
    )

    parser.add_argument(
        '--gain',
        type = int,
        help = 'Gain setting for all channels.  Valid settings {1, 2, 4, 6, 8, 12, 24}',
        default = 24
    )

    parser.add_argument(
        '--bias',
        type = str,
        help = 'Include channels in bias calculation',
        default = '11111111'
    )

    parser.add_argument(
        '--powerdown',
        type = str,
        help = 'Channels to disconnect/powerdown',
        default = '00011111'
    )

    parser.add_argument(
        '--impedance',
        action = 'store_true',
        help = "Enable continuous impedance monitoring",
        default = False
    )

    ## Decoder Arguments
    parser.add_argument(
        '--classes',
        type = int,
        help = 'Number of classes in decoder.  Ignored if --model specified',
        default = 2
    )

    parser.add_argument(
        '--model',
        type = lambda x: Path( x ).absolute(),
        help = 'Path to pre-trained model file',
        default = None
    )

    parser.add_argument(
        '--output',
        type = lambda x: Path( x ).absolute(),
        help = 'Directory to save output files',
        default = Path( '.' ) / "recordings"
    )

    args = parser.parse_args()

    gain_map: Dict[ int, GainState ] = {
        1:  GainState.GAIN_1,
        2:  GainState.GAIN_2,
        4:  GainState.GAIN_4,
        6:  GainState.GAIN_6,
        8:  GainState.GAIN_8,
        12: GainState.GAIN_12,
        24: GainState.GAIN_24
    }

    ch_setting = lambda ch_idx: ( 
        OpenBCIChannelSetting(
            gain = gain_map[ args.gain ], 
            power = ( PowerStatus.POWER_OFF 
                if args.powerdown[ch_idx] == '1' 
                else PowerStatus.POWER_ON ),
            bias = ( BiasSetting.INCLUDE   
                if args.bias[ch_idx] == '1'
                else BiasSetting.REMOVE 
            )
        )
    )

    settings = HololightSystemSettings(
        openbcisource_settings = OpenBCISourceSettings(
            device = args.device,
            blocksize = args.blocksize,
            impedance = args.impedance,
            ch_config = OpenBCIChannelConfigSettings(
                ch_setting = tuple( [ 
                    ch_setting( i ) for i in range( 8 ) 
                ] )
            )
        ),
        decoder_settings = ShallowFBCSPDecoderSettings(
            model_file = args.model
        ),
        modeltraining_settings = ModelTrainingSettings(
            settings = ModelTrainingLogicSettings(
                recording_dir = args.output 
            ),
            testsignal_settings = TestSignalInjectorSettings(
                enabled = True if args.device == 'simulator' else False
            ),
            websocket_settings = WebsocketSettings(
                host = "0.0.0.0",
                port = 8082
            )
        )
    )

    system = HololightSystem( settings )
    ez.run_system( system )
