from dataclasses import field

import panel
import param
import ezmsg.core as ez

import numpy as np

from ezmsg.eeg.eegmessage import EEGMessage

import matplotlib
matplotlib.use( "agg" )

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from typing import Optional, List

class EEGPlot( param.Parameterized ):
    # Reactive Parameters
    gain = param.Number( default = 1e6 )
    offset = param.Number( default = 2.0 )
    msg: Optional[ EEGMessage ] = param.Parameter()

    # Internal plotting functionality
    fig: Figure
    ax: Axes
    lines: Optional[ List[ Line2D ] ] = None

    def __init__( self, **params ) -> None:
        super().__init__( **params )
        self.fig, self.ax = plt.subplots( figsize = ( 8.0, 4.0 ) )

    def view( self ) -> Figure:
        if self.msg is not None:
            y_offsets = np.arange( self.msg.n_ch ) * self.offset
            arr = ( self.msg.data * self.gain ) + y_offsets
            eeg_dur = self.msg.n_time / self.msg.fs

            if self.lines is None:
                time = ( np.arange( self.msg.n_time ) / self.msg.fs ) - eeg_dur
                self.lines = self.ax.plot( time, arr )
            else:
                self.ax.set_autoscale_on( False )
                for idx, line in enumerate( self.lines ):
                    line.set_ydata( arr[ :, idx ] )

            self.ax.set_ylim( -self.offset, self.msg.n_ch * self.offset )
            self.ax.set_yticks( y_offsets )
            if self.msg.ch_names is not None:
                self.ax.set_yticklabels( self.msg.ch_names )
            self.ax.set_xlim( -eeg_dur, 0.0 )
            self.ax.grid( 'True' )

            self.ax.spines[ 'right' ].set_visible( False )
            self.ax.spines[ 'top' ].set_visible( False )

            self.fig.canvas.draw()


        return self.fig

    def panel( self ) -> panel.viewable.Viewable:
        return panel.Row(
            self.view,
            panel.Column(
                "<br>\n# EEG Plot",
                panel.widgets.NumberInput.from_param( self.param[ 'gain' ] ),
                panel.widgets.NumberInput.from_param( self.param[ 'offset' ] )
            )
        )

class EEGPlotterSettings( ez.Settings ):
    ...

class EEGPlotterState( ez.State ):
    plot: EEGPlot = field( default_factory = EEGPlot )

class EEGPlotter( ez.Unit ):

    SETTINGS: EEGPlotterSettings
    STATE: EEGPlotterState

    INPUT_SIGNAL = ez.InputStream( EEGMessage )

    @ez.subscriber( INPUT_SIGNAL )
    async def on_signal( self, msg: EEGMessage ) -> None:
        self.STATE.plot.msg = msg

    @ez.main
    def serve_dashboard( self ) -> None:
        panel.serve( self.STATE.plot.panel, port = 8082 )