import ezmsg as ez

from ezbci.openbci.openbci import OpenBCISourceSettings

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
        default = 50
    )

    parser.add_argument(
        '--poll',
        type = float,
        help = 'Poll Rate (Hz). 0 for auto-config',
        default = 0.0
    )

    args = parser.parse_args()

    openbcisource_settings = OpenBCISourceSettings(
        device = args.device,
        blocksize = args.blocksize,
        poll_rate = None if args.poll <= 0 else args.poll
    )

    settings = HololightSystemSettings(
        openbcisource_settings = openbcisource_settings
    )

    system = HololightSystem( settings )
    ez.run_system( system )
