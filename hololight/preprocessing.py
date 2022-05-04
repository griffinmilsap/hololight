import ezmsg.core as ez

from ezmsg.eeg.eegmessage import EEGMessage
from ezmsg.sigproc.decimate import Decimate, DownsampleSettings
from ezmsg.sigproc.butterworthfilter import ButterworthFilter, ButterworthFilterSettings
from ezmsg.sigproc.ewmfilter import EWMFilter, EWMFilterSettings
from ezmsg.sigproc.window import Window, WindowSettings

class PreprocessingSettings( ez.Settings ):
    # 1. Bandpass Filter
    bpfilt_order: int = 5
    bpfilt_cuton: float = 5.0 # Hz
    bpfilt_cutoff: float = 30.0 # Hz

    # 2. Downsample
    downsample_factor: int = 4 # Downsample factor to reduce sampling rate to ~ 100 Hz

    # 3. Exponentially Weighted Standardization
    ewm_history_dur: float = 2.0 # sec

    # 4. Sliding Window
    output_window_dur: float = 1.0 # sec
    # output_window_shift: float = 0.5 # sec
    output_window_shift: float = 1.0 # For training, we dont want overlap


class Preprocessing( ez.Collection ):
    """
    Preprocessing pipeline for an EEG neural network decoder.

    Preprocessing consists of:

    1. Bandpass Filtering
    X. TODO: Common Average Reference/Spatial Filtering
    2. Downsampling
    3. Exponentially Weighted Moving Standardization
    4. Windowing
    """

    SETTINGS: PreprocessingSettings

    INPUT_SIGNAL = ez.InputStream( EEGMessage )
    OUTPUT_SIGNAL = ez.OutputStream( EEGMessage )

    # Subunits
    BPFILT = ButterworthFilter()
    DECIMATE = Decimate()
    EWM = EWMFilter()
    WINDOW = Window()

    def configure( self ) -> None:
        self.BPFILT.apply_settings(
            ButterworthFilterSettings(
                order = self.SETTINGS.bpfilt_order,
                cuton = self.SETTINGS.bpfilt_cuton,
                cutoff = self.SETTINGS.bpfilt_cutoff
            )
        )
        self.DECIMATE.apply_settings( 
            DownsampleSettings(
                factor = self.SETTINGS.downsample_factor
            )
        )

        self.EWM.apply_settings(
            EWMFilterSettings(
                history_dur = self.SETTINGS.ewm_history_dur,
            )
        )

        self.WINDOW.apply_settings(
            WindowSettings(
                window_dur = self.SETTINGS.output_window_dur, # sec
                window_shift = self.SETTINGS.output_window_shift # sec
            )
        )


    def network( self ) -> ez.NetworkDefinition:
        return (
            ( self.INPUT_SIGNAL, self.BPFILT.INPUT_SIGNAL ),
            ( self.BPFILT.OUTPUT_SIGNAL, self.DECIMATE.INPUT_SIGNAL ),
            ( self.DECIMATE.OUTPUT_SIGNAL, self.EWM.INPUT_SIGNAL ),
            ( self.EWM.OUTPUT_SIGNAL, self.WINDOW.INPUT_SIGNAL ),
            ( self.WINDOW.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL )
        )
