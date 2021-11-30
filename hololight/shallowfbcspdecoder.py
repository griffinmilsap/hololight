from pathlib import Path

import numpy as np
import torch
import ezmsg as ez

from ezbci.eegmessage import EEGMessage, EEGInfoMessage, EEGDataMessage
from ezbci.stampedmessage import StampedMessage
from .shallowfbcspnet import ShallowFBCSPNet

from typing import ( 
    Optional,
    AsyncGenerator,
    Dict,
    Any
)

class DecoderOutput( StampedMessage ):
    output: np.ndarray # ( n_classes, ) vector of probabilities, dynamic

class ShallowFBCSPDecoderSettings( ez.Settings ):
    model_file: Optional[ Path ] = None

class ShallowFBCSPDecoderState( ez.State ):
    info: Optional[ EEGInfoMessage ] = None
    checkpoint: Optional[ Dict[ str, Any ] ] = None
    model: Optional[ torch.nn.Sequential ] = None

class ShallowFBSCPDecoder( ez.Unit ):
    """
    Performs inference on a pre-trained ShallowFBCSP Model
    """

    SETTINGS: ShallowFBCSPDecoderSettings
    STATE: ShallowFBCSPDecoderState

    INPUT_SIGNAL = ez.InputStream( EEGMessage )
    INPUT_MODEL = ez.InputStream( Path )
    OUTPUT_DECODE = ez.OutputStream( DecoderOutput )

    def initialize( self ) -> None:
        if self.SETTINGS.model_file is not None:
            self.load_model( self.SETTINGS.model_file )

    def load_model( self, model_file: Path ) -> None:
        self.STATE.checkpoint = torch.load( model_file, map_location = 'cpu' )
        self.STATE.model = ShallowFBCSPNet(
            self.STATE.checkpoint[ 'num_channels' ], 
            self.STATE.checkpoint[ 'num_classes' ],
            input_time_length = self.STATE.checkpoint[ 'num_time' ],
            final_conv_length = 'auto'
        ).construct()
        self.STATE.model.load_state_dict( self.STATE.checkpoint[ 'model_state_dict' ] )

    # We support dynamic model reload
    @ez.subscriber( INPUT_MODEL )
    async def on_model( self, message: Path ) -> None:
        self.load_model( message )

    @ez.subscriber( INPUT_SIGNAL )
    @ez.publisher( OUTPUT_DECODE )
    async def decode( self, message: EEGMessage ) -> AsyncGenerator:

        if isinstance( message, EEGInfoMessage ):
            self.STATE.info = message

        elif isinstance( message, EEGDataMessage ):
            if self.STATE.info is None: return
            if self.STATE.model is None: return

            # Perform Inference
            self.STATE.model.eval()

            with torch.no_grad():
                # Input to model is batch x ch x time x 1
                dim_order = ( self.STATE.info.ch_dim, self.STATE.info.time_dim )
                in_data: np.ndarray = np.transpose( message.data, dim_order )
                in_data = in_data[ np.newaxis, ..., np.newaxis ]
                in_tensor = torch.tensor( in_data.astype( np.float32 ), dtype = torch.float32 )

                output: torch.Tensor = self.STATE.model( in_tensor )
                out_probs = np.array( output.tolist() )[ 0, : ]

            yield( self.OUTPUT_DECODE, DecoderOutput( output = out_probs ) )

            





