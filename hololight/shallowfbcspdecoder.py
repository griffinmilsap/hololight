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
)

class DecoderOutput( StampedMessage ):
    output: np.ndarray # ( n_classes, ) vector of probabilities

class ShallowFBCSPDecoderSettings( ez.Settings ):
    """ 
    TODO: Once online training is implemented, make model_file required parameter
    TODO: Specify n_classes if model_file doesn't exist to train from scratch 
    For now, pre-trained model_file is optional.
    If we don't specify one, we don't perform inference.

    TODO: If model_file exists, n_classes is ignored.
    """
    model_file: Optional[ Path ] = None
    n_classes: Optional[ int ] = None

class ShallowFBCSPDecoderState( ez.State ):
    info: Optional[ EEGInfoMessage ] = None
    model: Optional[ torch.nn.Sequential ] = None

class ShallowFBSCPDecoder( ez.Unit ):
    """
    Performs inference on a pre-trained ShallowFBCSP Model.

    TODO: Implement online learning
    """

    SETTINGS: ShallowFBCSPDecoderSettings
    STATE: ShallowFBCSPDecoderState

    INPUT_SIGNAL = ez.InputStream( EEGMessage )
    OUTPUT_DECODE = ez.OutputStream( DecoderOutput )

    @ez.subscriber( INPUT_SIGNAL )
    @ez.publisher( OUTPUT_DECODE )
    async def decode( self, message: EEGMessage ) -> AsyncGenerator:

        if isinstance( message, EEGInfoMessage ):
            self.STATE.info = message

            build_model = lambda n_classes: ShallowFBCSPNet(
                self.STATE.info.n_ch, n_classes,
                input_time_length = self.STATE.info.n_time,
                final_conv_length = 'auto'
            ).construct()

            if self.SETTINGS.model_file is not None:
                checkpoint = torch.load( self.SETTINGS.model_file, map_location = 'cpu' )
                self.STATE.model = build_model( checkpoint[ 'num_classes' ] )
                self.STATE.model.load_state_dict( checkpoint[ 'model_state_dict' ] )

            elif self.SETTINGS.n_classes is not None:
                self.STATE.model = build_model( self.SETTINGS.n_classes )

        elif isinstance( message, EEGDataMessage ):
            if self.STATE.info is None: return
            if self.STATE.model is None: return

            # Perform Inference
            self.STATE.model.eval()

            with torch.no_grad():
                # batch x ch x time x 1
                dim_order = ( self.STATE.info.ch_dim, self.STATE.info.time_dim )
                in_data: np.ndarray = np.transpose( message.data, dim_order )
                in_data = in_data[ np.newaxis, ..., np.newaxis ]
                in_tensor = torch.tensor( in_data.astype( np.float32 ), dtype = torch.float32 )

                output: torch.Tensor = self.STATE.model( in_tensor )
                out_probs = np.array( output.tolist() )[ 0, : ]

            yield( self.OUTPUT_DECODE, DecoderOutput( output = out_probs ) )

            





