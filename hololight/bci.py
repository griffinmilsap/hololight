from pathlib import Path

import ezmsg as ez

from ezbci.openbci.openbci import OpenBCISourceSettings

from .shallowfbcspdecoder import ShallowFBCSPDecoderSettings
from .hololightsystem import HololightSystem, HololightSystemSettings

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description = 'Hololight ARBCI Lightbulb Demonstration'
    )

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
        '--poll',
        type = float,
        help = 'Poll Rate (Hz). 0 for auto-config',
        default = 0.0
    )

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

    args = parser.parse_args()

    openbcisource_settings = OpenBCISourceSettings(
        device = args.device,
        blocksize = args.blocksize,
        poll_rate = None if args.poll <= 0 else args.poll
    )

    decoder_settings = ShallowFBCSPDecoderSettings(
        n_classes = args.classes,
        model_file = args.model
    )

    settings = HololightSystemSettings(
        openbcisource_settings = openbcisource_settings,
        decoder_settings = decoder_settings
    )

    system = HololightSystem( settings )
    ez.run_system( system )
