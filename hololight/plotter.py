import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ezmsg.eeg.eegmessage import EEGMessage

import ezmsg as ez

from typing import (
    Optional
)

class PlotterSettings( ez.Settings ):
    rate: float = 10.0 # Hz

class PlotterState( ez.State ):
    data: Optional[ EEGMessage ] = None

class Plotter( ez.Unit ):

    SETTINGS: PlotterSettings
    STATE: PlotterState

    INPUT = ez.InputStream( EEGMessage )

    @ez.main
    def spawn_plot( self ) -> None:
        # Matplotlib requires blocking on the main thread... :( 
        # TODO: Consider moving self.ax to STATE.
        fig, self.ax = plt.subplots( dpi = 100, figsize = ( 4.0, 4.0 ) )
        anim = animation.FuncAnimation(
            fig, self._animate, interval = 1 / self.SETTINGS.rate * 1000
        )
        plt.show()

    def _animate( self, frame_idx: int ) -> None:
        if self.ax is None or self.STATE.data is None:
            return

        self.ax.clear()
        # self.ax.plot( ( self.STATE.data.data.T - self.STATE.data.data.mean( axis = 1 ) ).T )
        self.ax.plot( self.STATE.data.data )

        # self.ax.set_ylim( -2**23, 2**23 )

    @ez.subscriber( INPUT )
    async def plot( self, msg: EEGMessage ) -> None:
        self.STATE.data = msg