import ezmsg as ez

from .hololightsystem import HololightSystem, HololightSystemSettings

if __name__ == "__main__":
    import argparse

    # Parse arguments



    settings = HololightSystemSettings()
    system = HololightSystem( settings )
    ez.run_system( system )
