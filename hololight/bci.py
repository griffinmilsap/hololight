import time
from pathlib import Path

import ezmsg.core as ez

from ezmsg.eeg.openbci import (
    OpenBCISourceSettings, 
    GainState,
    PowerStatus,
    BiasSetting,
    OpenBCIChannelConfigSettings,
    OpenBCIChannelSetting,
)

from ezmsg.websocket import WebsocketSettings

from .go_task import GoTaskSettings
from .modeltraining import ( 
    ModelTrainingLogicSettings, 
    ModelTrainingSettings, 
    TestSignalInjectorSettings
)
from .shallowfbcspdecoder import ShallowFBCSPDecoderSettings
from .hololightsystem import HololightSystem, HololightSystemSettings
from .hololight import HololightSettings

from typing import (
    Dict,
    Optional
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
        help = 'Include channels in bias calculation. Default: 11111111',
        default = '11111111'
    )

    parser.add_argument(
        '--powerdown',
        type = str,
        help = 'Channels to disconnect/powerdown. Default: 00111111',
        default = '00111111'
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

    parser.add_argument(
        '--bridge',
        type = str,
        help = 'Hostname for Philips Hue Bridge',
        default = None
    )

    parser.add_argument(
        '--cert',
        type = lambda x: Path( x ),
        help = "Certificate file for frontend server",
        default = ( Path( '.' ) / 'cert.pem' ).absolute()
    )

    args = parser.parse_args()

    device: str = args.device
    blocksize: int = args.blocksize
    gain: int = args.gain
    bias: str = args.bias
    powerdown: str = args.powerdown
    impedance: bool = args.impedance

    classes: int = args.classes
    model: Optional[ Path ] = args.model
    output: Path = args.output
    
    bridge: Optional[ str ] = args.bridge
    cert: Path = args.cert

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
            gain = gain_map[ gain ], 
            power = ( PowerStatus.POWER_OFF 
                if powerdown[ch_idx] == '1' 
                else PowerStatus.POWER_ON ),
            bias = ( BiasSetting.INCLUDE   
                if bias[ch_idx] == '1'
                else BiasSetting.REMOVE 
            )
        )
    )

    if not cert.exists():
        raise ValueError( f"Certificate {cert} does not exist" )

    settings = HololightSystemSettings(

        openbcisource_settings = OpenBCISourceSettings(
            device = device,
            blocksize = blocksize,
            impedance = impedance,
            ch_config = OpenBCIChannelConfigSettings(
                ch_setting = tuple( [ 
                    ch_setting( i ) for i in range( 8 ) 
                ] )
            )
        ),

        decoder_settings = ShallowFBCSPDecoderSettings(
            model_file = model
        ),

        modeltraining_settings = ModelTrainingSettings(

            settings = ModelTrainingLogicSettings(
                recording_dir = output 
            ),

            testsignal_settings = TestSignalInjectorSettings(
                enabled = True if device == 'simulator' else False
            ),

            websocket_settings = WebsocketSettings(
                host = "0.0.0.0",
                port = 8083
            )
        ),

        demo_settings = HololightSettings(
            cert = cert,
            bridge_host = bridge
        )
    )

    system = HololightSystem( settings )
    ez.run_system( system )
