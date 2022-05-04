import ezmsg.core as ez

from serialize import Deserialize
from plotter import Plotter

from ezmsg.websocket import WebsocketClient, WebsocketSettings

class JoyRecorderVizSettings( ez.Settings ):
    client_settings: WebsocketSettings

class JoyRecorderVizSystem( ez.System ):

    SETTINGS: JoyRecorderVizSettings

    # Subunits
    CLIENT = WebsocketClient()
    DESERIALIZE = Deserialize()
    PLOTTER = Plotter()

    def configure( self ) -> None:
        self.CLIENT.apply_settings( 
            self.SETTINGS.client_settings 
        )

    def network( self ) -> ez.NetworkDefinition:
        return (
            ( self.CLIENT.OUTPUT, self.DESERIALIZE.INPUT_BYTES ),
            ( self.DESERIALIZE.OUTPUT, self.PLOTTER.INPUT )
        )

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description = 'Live Visualizer for Joystick Recorder Task'
    )

    parser.add_argument(
        '--host',
        type = str,
        help = 'Host for websocket',
        default = '127.0.0.1'
    )

    parser.add_argument(
        '--port',
        type = int,
        help = 'Port for websocket',
        default = 23456
    )

    args = parser.parse_args()

    settings = JoyRecorderVizSettings(
        client_settings = WebsocketSettings(
            host = args.host,
            port = args.port
        )
    )

    system = JoyRecorderVizSystem( settings )

    ez.run_system( system )

