import json
import time
import base64
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import (
    Dataset,
    DataLoader, 
    random_split, 
    RandomSampler
)
from torch.optim.lr_scheduler import CosineAnnealingLR

from .shallowfbcspnet import ShallowFBCSPNet

from typing import (
    Optional,
    List,
    Tuple
)

class HoloLightDataset( Dataset ):
    
    fs: int
    ch_names: List[ str ]
    shape: Tuple[ int, ... ]
    trials: torch.tensor
    labels: torch.tensor

    def __init__( self, input_dir: Path ):
        
        cur_class: Optional[ int ] = None

        self.trials = []
        self.labels = []

        for data_file in input_dir.glob( '*.txt' ):
            with open( data_file, 'r' ) as data_f:
                for line in data_f:
                    msg = json.loads( line )

                    if msg[ '_type' ] == 'EEGInfoMessage':
                        # Assumes all files have same fs, ch_names and shape
                        self.fs = msg[ 'fs' ]
                        self.ch_names = msg[ 'ch_names' ]
                        self.shape = msg[ 'shape' ]

                    elif msg[ '_type' ] == 'EEGDataMessage':
                        if cur_class is not None:
                            dtype, data, shape = tuple( msg[ 'data' ] )
                            b64_data = base64.b64decode( data.encode( 'ascii' ) )
                            block = np.frombuffer( b64_data, dtype = dtype ).reshape( shape )
                            self.trials.append( torch.tensor( block.T, dtype = torch.float32 ) )
                            self.labels.append( cur_class )

                    elif msg[ '_type' ] == 'GoTaskMessage':
                        cur_class = msg[ 'trial_class' ] if msg[ 'stage' ] == 'ACTIVITY' else None
                        
        self.trials = torch.stack( self.trials )
        self.labels = torch.tensor( self.labels )
                        
    def __len__( self ) -> int:
        return len( self.labels )

    def __getitem__( self, idx ) -> Tuple[ torch.tensor, int ]:
        return self.trials[ idx, ... ], self.labels[ idx, ... ]
            

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser( 
        description = 'ShallowFBCSP Training Script'
    )

    parser.add_argument(
        '--dir',
        type = lambda s: Path( s ).absolute(),
        help = 'Directory of input files for training'
    )

    parser.add_argument(
        '--tag',
        type = str,
        help = 'Output training tag',
        default = None
    )

    parser.add_argument( 
        '--progress',
        action = 'store_true',
        help = 'Output a progress bar with tqdm',
        default = False
    )

    training_group = parser.add_argument_group( 'training' )

    training_group.add_argument(
        '--batchsize',
        type = int,
        help = "Batch size for training",
        default = 64
    )

    training_group.add_argument(
        '--epochs',
        type = int,
        help = 'Number of training epochs',
        default = 100
    )

    training_group.add_argument( 
        '--lr',
        type = float,
        help = 'Learning rate for training (AdamW)',
        default = 0.00625
    )

    training_group.add_argument(
        '--ratio',
        type = float,
        help = "Percentage of trials to use for train split",
        default = 0.9
    )

    model_group = parser.add_argument_group( 'model' )

    model_group.add_argument(
        '--timefilters',
        type = int,
        help = "Number of time kernels (FIR Filters) to fit",
        default = 40
    )

    model_group.add_argument(
        '--timefilterlength',
        type = int,
        help = "Size of time kernels in samples -- akin to FIR filter order",
        default = 25
    )

    model_group.add_argument(
        '--spatfilters',
        type = int,
        help = "Number of spatial kernels (Spatial filters) to fit",
        default = 40
    )

    model_group.add_argument(
        '--pooltimelength',
        type = int,
        help = "Mean pooling for time dimension in samples",
        default = 75
    )

    model_group.add_argument(
        '--pooltimestride',
        type = int,
        help = "Mean pooling stride for time dimension (internal downsampling)",
        default = 15
    )

    args = parser.parse_args()

    input_dir = args.dir
    tag = time.strftime( '%Y%m%dT%H%M%S' ) if args.tag is None else args.tag

    progress = args.progress # True

    train_ratio = args.ratio # 0.9
    learning_rate = args.lr # 0.00625
    max_epochs = args.epochs # 100
    batch_size = args.batchsize # 64

    n_filters_time = args.timefilters # 5
    n_filters_spat = args.spatfilters # 5
    pool_time_length = args.pooltimelength # 25
    pool_time_stride = args.pooltimestride # 5
    filter_time_length = args.timefilterlength # 10

    ## Build model
    dset = HoloLightDataset( input_dir )
    train_dset_size = int( len( dset ) * train_ratio )
    train_dset, test_dset = random_split( dset, [ train_dset_size, len( dset ) - train_dset_size ] )

    model_definition = ShallowFBCSPNet( 
        dset.shape[1], len( np.unique( dset.labels ) ), 
        input_time_length = dset.shape[0], 
        final_conv_length = 'auto',
        split_first_layer = False,
        filter_time_length = filter_time_length,
        n_filters_time = n_filters_time,
        n_filters_spat = n_filters_spat,
        pool_time_length = pool_time_length,
        pool_time_stride = pool_time_stride
    )

    model = model_definition.construct()

    model_parameters = filter( lambda p: p.requires_grad, model.parameters() )
    params = sum( [ np.prod( p.size() ) for p in model_parameters ] )
    # print( f'Model has {params} trainable parameters' )

    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.AdamW( 
        model.parameters(), 
        lr = learning_rate, 
        weight_decay = 0.1 
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max = max_epochs / 4 )

    best_loss = None
    best_loss_epoch = None

    train_loss, test_loss, test_accuracy = [], [], []
    lr = []

    epoch_itr = range( max_epochs )

    if progress:
        try: 
            from tqdm.autonotebook import tqdm
            epoch_itr = tqdm( epoch_itr )
        except:
            warnings.warn( 'tqdm not installed, no progressbar will show.' )
            progress = False
        
    for epoch in epoch_itr:

        model.train()
        train_loss_batches = []
        for train_feats, train_labels in DataLoader(
            train_dset, batch_size = batch_size, 
            sampler = RandomSampler( train_dset )
        ):
            pred = model( train_feats )
            loss = loss_fn( pred, train_labels )
            train_loss_batches.append( loss.item() )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        scheduler.step()
            
        lr.append( scheduler.get_last_lr()[0] )
        train_loss.append( np.mean( train_loss_batches ) )

        model.eval()
        with torch.no_grad():
            accuracy = 0
            test_loss_batches = []
            for test_feats, test_labels in DataLoader(
                test_dset, batch_size = batch_size, 
                sampler = RandomSampler( test_dset )
            ):
                output = model( test_feats )
                test_loss_batches.append( loss_fn( output, test_labels ).item() )
                accuracy += ( output.argmax( axis = 1 ) == test_labels ).sum().item()

            test_loss.append( np.mean( test_loss_batches ) )
            test_accuracy.append( accuracy / len( test_dset ) )
            
    acc_str = f'Accuracy: {test_accuracy[-1] * 100.0:0.2f}%'

    fig, ax = plt.subplots( dpi = 100, figsize = ( 6.0, 4.0 ) )
    ax.plot( train_loss, label = 'Train' )
    ax.plot( test_loss, label = 'Test' )
    ax.plot( test_accuracy, label = 'Test Accuracy' )
    ax.plot( lr, label = 'Learning Rate' )
    ax.legend()
    ax.set_yscale( 'log' )
    ax.set_xlabel( 'Epoch' )

    out_train = input_dir / f'{tag}_train.png'
    fig.savefig( out_train )

    model.eval()

    output = [ 
        ( model( test_feats ).argmax( axis = 1 ), test_labels )
        for test_feats, test_labels 
        in DataLoader( test_dset, batch_size = batch_size ) 
    ]

    decode, test_y = zip( *output )
    test_y = torch.cat( test_y, axis = 0 )
    decode = torch.cat( decode, axis = 0 )

    classes = np.unique( dset.labels )
    confusion = np.zeros( ( len( classes ), len( classes ) ) )
    for true_idx, true_class in enumerate( classes ):
        class_trials = np.where( test_y == true_class )[0]
        for pred_idx, pred_class in enumerate( classes ):
            num_preds = ( decode[ class_trials ] == pred_class ).sum().item()
            confusion[ true_idx, pred_idx ] = num_preds / len( class_trials )
    

    fig, ax = plt.subplots( dpi = 100 )
    corners = np.arange( len( classes ) + 1 ) - 0.5
    im = ax.pcolormesh( 
        corners, corners, confusion, alpha = 0.5,
        cmap = plt.cm.Blues, vmin = 0.0, vmax = 1.0
    )

    for row_idx, row in enumerate( confusion ):
        for col_idx, freq in enumerate( row ):
            ax.annotate( 
                str( freq ), ( col_idx, row_idx ), 
                ha = 'center', va = 'center' 
            )

    ax.set_aspect( 'equal' )
    ax.set_xticks( classes )
    ax.set_yticks( classes )
    ax.set_ylabel( 'True Class' )
    ax.set_xlabel( 'Predicted Class' )
    ax.invert_yaxis( )
    fig.colorbar( im )
    ax.set_title( acc_str )

    out_accuracy = input_dir / f'{tag}_acc.png'
    fig.savefig( out_accuracy )

    checkpoint = {
        'model_definition': model_definition,
        'fs': dset.fs, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    out_checkpoint = input_dir / f'{tag}.checkpoint'
    torch.save( checkpoint, out_checkpoint )