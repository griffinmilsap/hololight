# Note: This whole model/code was torn out of braindecode
# https://braindecode.org/

from dataclasses import dataclass, field

import torch as th
import numpy as np
from torch import nn
from torch.nn import init

from typing import (
    Optional,
    Callable,
    Union
)

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
                self.expression_fn.func.__name__, str( self.expression_fn.kwargs )
            )
        elif hasattr( self.expression_fn, "__name__" ):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr( self.expression_fn )
        return f'{ self.__class__.__name__ }(expression={ str( expression_str ) })'
    
class Ensure4d( nn.Module ):
    def forward( self, x ):
        while( len( x.shape ) < 4 ):
            x = x.unsqueeze( -1 )
        return x

# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output( x: th.Tensor ) -> th.Tensor:
    assert x.size()[ 3 ] == 1
    x = x[ :, :, :, 0 ]
    if x.size()[ 2 ] == 1:
        x = x[ :, :, 0 ]
    return x

def _transpose_time_to_spat( x: th.Tensor ) -> th.Tensor:
    return x.permute( 0, 3, 2, 1 )

@dataclass
class ShallowFBCSPNet:

    """
    Shallow ConvNet model from [2]_.

    Input is ( batch x channel x time x 1 )
    Output is ( batch x class ) probabilities 0-1

    References
    ----------

    .. [2] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    in_chans: int
    n_classes: int
    input_time_length: Optional[ int ] = None
    n_filters_time: int = 40
    filter_time_length: int = 25
    n_filters_spat: int = 40
    pool_time_length: int = 75
    pool_time_stride: int = 15
    final_conv_length: Union[ str, int ] = 30 # Could also be 'auto'
    conv_nonlin: Callable = field( default_factory = lambda: square )
    pool_mode: str = 'mean'
    pool_nonlin: Callable = field( default_factory = lambda: safe_log )
    split_first_layer: bool = True
    batch_norm: bool = True
    batch_norm_alpha: float = 0.1
    drop_prob: float = 0.5

    def construct( self ) -> th.nn.Sequential:

        if self.final_conv_length == "auto":
            assert self.input_time_length is not None

        pool_class = dict( max = nn.MaxPool2d, mean = nn.AvgPool2d )[ self.pool_mode ]
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

        model.add_module( "conv_nonlin", Expression( self.conv_nonlin ) )

        model.add_module( "pool",
            pool_class(
                kernel_size = ( self.pool_time_length, 1 ),
                stride = ( self.pool_time_stride, 1 ),
            ),
        )

        model.add_module( "pool_nonlin", Expression( self.pool_nonlin ) )
        model.add_module( "drop", nn.Dropout( p = self.drop_prob ) )
        model.eval()

        if self.final_conv_length == "auto":
            out: th.Tensor = model(
                th.tensor(
                    np.ones(
                        ( 1, self.in_chans, self.input_time_length, 1 ),
                        dtype = np.float32,
                    ),
                    dtype = th.float32
                )
            )
            # n_out_time = out.cpu().data.numpy().shape[2]
            n_out_time = np.array( out.tolist() ).shape[2]
            self.final_conv_length = n_out_time

        model.add_module( "conv_classifier",
            nn.Conv2d(
                n_filters_conv,
                self.n_classes,
                ( self.final_conv_length, 1 ),
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

        return model
