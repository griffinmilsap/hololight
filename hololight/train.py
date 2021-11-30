import json
import time
import base64

from pathlib import Path

import xarray as xr
import numpy as np

import torch

from shallowfbcspnet import ShallowFBCSPNet

def load_file( data_file: Path ) -> xr.Dataset:

    fs = None
    ch_names = None

    eeg_blocks = []
    eeg_timestamps = []

    events = []
    event_timestamps = []
    
    with open( data_file, 'r' ) as data_f:
        for line in data_f:
            msg = json.loads( line )
            if msg[ '_type' ] == 'EEGInfoMessage':
                fs = msg[ 'fs' ]
                ch_names = msg[ 'ch_names' ]

            elif msg[ '_type' ] == 'EEGDataMessage':
                dtype, data, shape = tuple( msg[ 'data' ] )
                b64_data = base64.b64decode( data.encode( 'ascii' ) )
                block = np.frombuffer( b64_data, dtype = dtype ).reshape( shape )
                eeg_blocks.append( block )
                eeg_timestamps.append( msg[ '_timestamp' ] )

            elif msg[ '_type' ] == 'GoTaskMessage':
                events.append( dict(
                    stage = msg[ 'stage' ],
                    trial_class = msg[ 'trial_class' ],
                ) )
                event_timestamps.append( msg[ '_timestamp' ] )

    blocks = xr.DataArray( 
        np.array( eeg_blocks ), 
        name = "eeg",
        dims = [ 'block', 'element', 'channel' ], 
        coords = dict(
            _timestamp = ( 'block', eeg_timestamps ),
            ch_name = ( 'channel', ch_names )
        ),
        attrs = dict( fs = fs )
    )

    events = xr.DataArray( 
        events, 
        name = 'events', 
        dims = [ 'time' ], 
        coords = dict( 
            time = ( 'time', np.array( event_timestamps ) ) 
        ) 
    )

    eeg = blocks.stack( time = [ 'block', 'element' ] )
    eeg = eeg.reset_index( 'time' )

    eeg._timestamp.loc[ dict( 
        time = eeg.element != len( blocks.element ) - 1 
    ) ] = np.nan
    time_coords = eeg._timestamp.interpolate_na( 
        dim = 'time', 
        fill_value = 'extrapolate' 
    )
    eeg = eeg \
        .assign_coords( 
            time = time_coords,
            sample = ( 'time', np.arange( len( eeg.time ) ).astype( np.int64 ) )
        ) \
        .drop( '_timestamp' ) \
        .assign_attrs( fs = fs )
    
    ds = xr.Dataset( dict( 
            eeg = eeg, 
            events = events 
        ),
        attrs = dict(
            filename = data_file.stem,
        )
    )

    return ds

def split( ds, ratio, shuffle = True ):
    
    trials = ds.trials
    labels = ds.labels
    
    trial_dim = labels.dims[0]

    classes, counts = np.unique( labels, return_counts = True )
    trials_per_class = np.min( counts )
    split_stop = int( trials_per_class * ratio )
    indices = [ labels.trial.where( labels == c ).dropna( 'trial' ).copy().values.astype( np.int64 ) for c in classes ]

    a, b = [], []
    for i, c in zip( indices, classes ):
        if shuffle: np.random.shuffle( i )
        a.append( i[ : split_stop ] )
        b.append( i[ split_stop : trials_per_class ] )
        
    a = np.array( a ).flatten()
    b = np.array( b ).flatten()
        
    if shuffle: 
        np.random.shuffle( a )
        np.random.shuffle( b )
        
    return ( 
        ds.isel( { trial_dim: np.array( a ).flatten() } ), 
        ds.isel( { trial_dim: np.array( b ).flatten() } )
    )

if __name__ == '__main__':

    import argparse

    try:
        import matplotlib.pyplot as plt
        plot = True
    except ImportError:
        plot = False


    # TODO: Argparse
    rec_path = Path( 'gm_motor_both_hands' )
    batch_size = 4
    max_epochs = 5000
    no_improvement_epochs = 100
    learning_rate = 0.000625
    save_output = True

    training_timestamp = time.strftime( '%Y%m%dT%H%M%S' )

    dataset = []
    for data_file in rec_path.glob( '*.txt' ):

        rec = load_file( data_file )
        eeg = rec.eeg.dropna( 'time' )
        events = rec.events.dropna( 'time' )

        trials, labels = [], []
        for t, event_da in events.groupby( 'time' ):
            event = event_da.item()
            if event[ 'stage' ] != 'ACTIVITY': continue
            t_idx = eeg.indexes[ 'time' ].get_loc( t + 0.5, method = 'nearest' )
            trials.append( eeg.isel( time = slice( t_idx, t_idx + int( eeg.fs ) ) ).drop( 'time') )
            labels.append( event[ 'trial_class' ] )
            
        dataset.append( 
            xr.Dataset( dict( 
                trials = xr.concat( trials, dim = 'trial' ), 
                labels = xr.DataArray( labels, dims = [ 'trial' ] ) 
        ) ) )

    dataset = xr.concat( dataset, dim = 'trial' )   

    train_dset, test = split( dataset, 0.8 )
    train, valid = split( train_dset, 0.75 )

    classes = np.unique( dataset.labels )
    num_classes = len( classes ) 

    model = ShallowFBCSPNet( 
        len( dataset.channel ), num_classes, 
        input_time_length = len( dataset.time ), 
        final_conv_length = 'auto' 
    ).construct()

    dim_order = [ 'trial', 'channel', 'time' ]
    X_tensor = lambda da: torch.tensor( 
        da.transpose( 
            *[ d for d in dim_order if d in da.dims ], 
            transpose_coords = False 
        ).values.astype( np.float32 )[ ..., np.newaxis ] 
    )

    y_tensor = lambda da: torch.tensor( da.values.astype( np.int64 ) )

    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.AdamW( 
        model.parameters(), 
        lr = learning_rate, 
        weight_decay = 0 
    )

    valid_X, test_X = tuple( [ X_tensor( da ) for da in [ valid.trials, test.trials ] ] )
    valid_y, test_y = tuple( [ y_tensor( da ) for da in [ valid.labels, test.labels ] ] )

    best_loss = None
    best_loss_epoch = None

    cur_train_data = train
    train_loss, valid_loss, test_loss = [], [], []

    for epoch in range( max_epochs ):

        model.train()
        for bounds, batch_ds in cur_train_data.groupby_bins( 
            'trial', bins = len( train.trial ) // batch_size 
        ):
            train_X = X_tensor( batch_ds.trials )
            train_y = y_tensor( batch_ds.labels )

            pred = model( train_X )
            loss = loss_fn( pred, train_y )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss.append( loss.item() )

        model.eval()
        with torch.no_grad():
            valid_loss.append( loss_fn( model( valid_X ), valid_y ).item() )
            test_loss.append( loss_fn( model( test_X ), test_y ).item() )

            loss = valid_loss[-1] if cur_train_data is train else test_loss[-1]

            best_loss = loss if best_loss is None else ( 
                loss if loss < best_loss else best_loss 
            )

            if best_loss is loss:
                best_loss_epoch = epoch
                checkpoint = {
                    'epoch': epoch,
                    'num_classes': num_classes,
                    'num_channels': len( dataset.channel ),
                    'num_time': len( dataset.time ),
                    'fs': eeg.fs, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss[-1],
                    'valid_loss': valid_loss[-1],
                    'test_loss': test_loss[-1]
                }

            elif epoch > ( best_loss_epoch + no_improvement_epochs ):
                print( 'Completed Training Pass' )

                # Load best checkpoint
                model.load_state_dict( checkpoint[ 'model_state_dict' ] )
                optimizer.load_state_dict( checkpoint[ 'optimizer_state_dict' ] )

                # Add validation data and train
                if cur_train_data is train:
                    cur_train_data = train_dset
                    best_loss = None
                    best_loss_epoch = None
                else:
                    print( 'Done Training' )
                    break

    out_checkpoint = rec_path / f'{training_timestamp}.checkpoint'
    if save_output: 
        torch.save( checkpoint, out_checkpoint )

    model.eval()
    decode = model( test_X ).argmax( axis = 1 )

    ## Present some training information/accuracy
    accuracy = ( decode == test_y ).sum().item() / len( test_y )
    acc_str = f'Accuracy: {accuracy * 100.0:0.2f}%'

    print( acc_str )

    confusion = np.zeros( ( num_classes, num_classes ) )
    for true_idx, true_class in enumerate( classes ):
        class_trials = np.where( test_y == true_class )[0]
        for pred_idx, pred_class in enumerate( classes ):
            num_preds = ( decode[ class_trials ] == pred_class ).sum().item()
            confusion[ true_idx, pred_idx ] = num_preds / len( class_trials )
            
    print( confusion )

    if plot:

        fig, ax = plt.subplots( dpi = 100 )
        ax.plot( np.array( train_loss ), label = 'train' )
        ax.plot( np.array( valid_loss ), label = 'valid' )
        ax.plot( np.array( test_loss ), label = 'test' )

        ax.set_ylabel( 'loss' )
        ax.set_xlabel( 'iteration' )
        ax.legend()
        ax.set_yscale( 'log' )

        if save_output: 
            fig.savefig( rec_path / f'{training_timestamp}_training.png' )

    if plot:
        fig, ax = plt.subplots( dpi = 100 )
        corners = np.arange( num_classes + 1 ) - 0.5
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
        fig.colorbar( im )
        ax.set_title( acc_str )

        if save_output: 
            fig.savefig( rec_path / f'{training_timestamp}_confusion.png' )