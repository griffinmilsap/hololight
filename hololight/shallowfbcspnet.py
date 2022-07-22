# Note: This whole code was adapted from braindecode ShallowFBCSP
# https://braindecode.org/

from dataclasses import dataclass

import torch as th
import numpy as np
from torch import nn
from torch.nn import init

from typing import Optional, Union, Callable, Tuple

def square( x: th.Tensor ) -> th.Tensor:
    return x * x

def safe_log( x: th.Tensor, eps: float = 1e-6 ) -> th.Tensor:
    """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return th.log( th.clamp( x, min = eps ) )

class Expression( th.nn.Module ):
    """
    Compute given expression on forward pass.
    Parameters
    ----------
    expression_fn: function
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    expression_fn: Callable

    def __init__( self, expression_fn ):
        super( Expression, self ).__init__()
        self.expression_fn = expression_fn

    def forward( self, *args ):
        return self.expression_fn( *args )

    def __repr__( self ):
        if hasattr( self.expression_fn, "func" ) and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str( self.expression_fn.kwargs ) # noqa
            )
        elif hasattr( self.expression_fn, "__name__" ):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr( self.expression_fn )
        return f'{ self.__class__.__name__ }(expression={ str( expression_str ) })'
    
class Ensure4d( nn.Module ):
    def forward( self, x: th.Tensor ):
        while( len( x.shape ) < 4 ):
            x = x.unsqueeze( -1 )
        return x

def to_dense_prediction_model( 
    model: nn.Module, 
    axis: Union[ int, Tuple[ int, int ] ] = ( 2, 3 ) 
) -> None:
    """
    Transform a sequential model with strides to a model that outputs
    dense predictions by removing the strides and instead inserting dilations.
    Modifies model in-place.
    Parameters
    ----------
    model: torch.nn.Module
        Model which modules will be modified
    axis: int or (int,int)
        Axis to transform (in terms of intermediate output axes)
        can either be 2, 3, or (2,3).
    Notes
    -----
    Does not yet work correctly for average pooling.
    Prior to version 0.1.7, there had been a bug that could move strides
    backwards one layer.
    """
    if not hasattr( axis, "__len__" ):
        axis = [ axis ]
    assert all( [ ax in [ 2, 3 ] for ax in axis ] ), "Only 2 and 3 allowed for axis"
    axis = np.array( axis ) - 2
    stride_so_far = np.array( [ 1, 1 ] )
    for module in model.modules():
        if hasattr( module, "dilation" ):
            assert module.dilation == 1 or ( module.dilation == ( 1, 1 ) ), (
                "Dilation should equal 1 before conversion, maybe the model is "
                "already converted?"
            )
            new_dilation = [ 1, 1 ]
            for ax in axis:
                new_dilation[ ax ] = int( stride_so_far[ ax ] )
            module.dilation = tuple( new_dilation )
        if hasattr( module, "stride" ):
            if not hasattr( module.stride, "__len__" ):
                module.stride = ( module.stride, module.stride )
            stride_so_far *= np.array( module.stride )
            new_stride = list( module.stride )
            for ax in axis:
                new_stride[ ax ] = 1
            module.stride = tuple( new_stride )

def get_output_shape( model: nn.Module, in_chans: int, input_window_samples: int ) -> th.Size:
    """Returns shape of neural network output for batch size equal 1.
    Returns
    -------
    output_shape: tuple
        shape of the network output for `batch_size==1` (1, ...)
    """
    with th.no_grad():
        dummy_input = th.ones(
            1, in_chans, input_window_samples,
            dtype = next( model.parameters() ).dtype,
            device = next( model.parameters() ).device,
        )
        output_shape = model( dummy_input ).shape
    return output_shape

# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output( x: th.Tensor ) -> th.Tensor:
    assert x.size()[ 3 ] == 1
    x = x[ :, :, :, 0 ]
    return x

def _transpose_time_to_spat( x: th.Tensor ) -> th.Tensor:
    return x.permute( 0, 3, 2, 1 )

@dataclass
class ShallowFBCSPNet:

    """
    Shallow ConvNet model from [2]_.

    Input is ( batch x channel x time ) Standardized EEG timeseries ( N(0,1) across window )
    Output is ( batch x class x time ) log-probabilities
        To get probabilities, recommend np.exp( output ).mean( dim = 2 ).squeeze()

    Default parameters are fine-tuned for EEG classification of spectral modulation
    between 8 and 112 Hz on time-series multi-channel EEG data sampled at ~250 Hz
    Recommend training on 4 second windows (input_time_length = 1000 samples)

    If doing "cropped training", inferencing can happen on smaller temporal windows
    If not doing cropped training, inferencing must happen on same window size as training.

    References
    ----------

    .. [2] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    # IO Parameters
    in_chans: int
    n_classes: int

    # Training information
    input_time_length: int # samples
    cropped_training: bool = False

    # First step -- Temporal Convolution (think FIR filtering)
    n_filters_time: int = 40
    filter_time_length: int = 25 # samples (think FIR filter order)

    # Second step -- Common Spatial Pattern (spatial weighting (no conv))
    n_filters_spat: int = 40
    split_first_layer: bool = True # If False, smash temporal and spatial conv layers together

    # Third step -- Nonlinearities
    #   First Nonlinearity: Batch normalization centers data
    batch_norm: bool = True
    batch_norm_alpha: float = 0.1

    #   Second Nonlinearity: Conv Nonlinearity.  'square' extracts filter output power
    conv_nonlin: Optional[ str ] = 'square' # || safe_log; No nonlin if None

    # Fourth step - Temporal pooling.  Aggregates and decimates spectral power
    pool_time_length: int = 75 # samples (think low pass filter on spectral content)
    pool_time_stride: int = 15 # samples (think decimation of spectral content)
    pool_mode: str = 'mean' # || 'max'

    # Fifth Step - Pool Nonlinearity.  'safe_log' makes spectral power normally distributed
    pool_nonlin: Optional[ str ] = 'safe_log' # || square; No nonlin if None
    
    # Sixth Step - Dropout layer for training resilient network and convergence
    drop_prob: float = 0.5

    # Seventh step -- Dense layer to output class. No parameters
    # Eighth step -- LogSoftmax - Output to probabilities. No parameters

    def construct( self ) -> th.nn.Module:

        pool_class = dict( 
            max = nn.MaxPool2d, 
            mean = nn.AvgPool2d 
        )[ self.pool_mode ]

        nonlin_dict = dict( 
            square = square,
            safe_log = safe_log 
        )

        model = nn.Sequential()
        
        model.add_module( "ensuredims", Ensure4d() )

        if self.split_first_layer:
            model.add_module( "dimshuffle", Expression( _transpose_time_to_spat ) )

            model.add_module( "conv_time", 
                nn.Conv2d(
                    1,
                    self.n_filters_time,
                    ( self.filter_time_length, 1 ),
                    stride = 1,
                ),
            )
            model.add_module( "conv_spat",
                nn.Conv2d(
                    self.n_filters_time,
                    self.n_filters_spat,
                    ( 1, self.in_chans ),
                    stride = 1,
                    bias = not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_spat
        else:
            model.add_module( "conv_time",
                nn.Conv2d(
                    self.in_chans,
                    self.n_filters_time,
                    ( self.filter_time_length, 1 ),
                    stride = 1,
                    bias = not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_time

        if self.batch_norm:
            model.add_module( "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv, 
                    momentum = self.batch_norm_alpha, 
                    affine = True
                ),
            )

        if self.conv_nonlin:
            model.add_module( "conv_nonlin", Expression( 
                nonlin_dict[ self.conv_nonlin ] 
            ) )

        model.add_module( "pool",
            pool_class(
                kernel_size = ( self.pool_time_length, 1 ),
                stride = ( self.pool_time_stride, 1 ),
            ),
        )

        if self.pool_nonlin:
            model.add_module( "pool_nonlin", Expression( 
                nonlin_dict[ self.pool_nonlin ] 
            ) )

        model.add_module( "drop", nn.Dropout( p = self.drop_prob ) )
        # model.eval()

        n_out_time = get_output_shape( model, self.in_chans, self.input_time_length )[2]
        if self.cropped_training: n_out_time = int( n_out_time // 2 )

        model.add_module( "conv_classifier",
            nn.Conv2d(
                n_filters_conv,
                self.n_classes,
                ( n_out_time, 1 ),
                bias = True,
            ),
        )

        model.add_module( "softmax", nn.LogSoftmax( dim = 1 ) )
        model.add_module( "squeeze", Expression( _squeeze_final_output ) )

        # Initialization, xavier is same as in paper...
        init.xavier_uniform_( model.conv_time.weight, gain = 1 )

        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or ( not self.batch_norm ):
            init.constant_( model.conv_time.bias, 0 )
        if self.split_first_layer:
            init.xavier_uniform_( model.conv_spat.weight, gain = 1 )
            if not self.batch_norm:
                init.constant_( model.conv_spat.bias, 0 )
        if self.batch_norm:
            init.constant_( model.bnorm.weight, 1 )
            init.constant_( model.bnorm.bias, 0 )
        init.xavier_uniform_( model.conv_classifier.weight, gain = 1 )
        init.constant_( model.conv_classifier.bias, 0 )

        if self.cropped_training:
            to_dense_prediction_model( model )
            len_temporal_receptive_field = get_output_shape( 
                model, self.in_chans, self.input_time_length )[2]
            print( len_temporal_receptive_field )

        return model
