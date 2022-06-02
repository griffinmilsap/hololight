# hololight

## Setup

```
$ python3 -m venv env
$ source env/bin/activate
(env) $ pip install --upgrade pip
(env) $ pip install -e ../path/to/ezmsg
(env) $ pip install -e ../path/to/ezmsg/extensions/ezmsg-sigproc
(env) $ pip install -e ../path/to/ezmsg/extensions/ezmsg-eeg
(env) $ pip install -e ../path/to/ezmsg/extensions/ezmsg-websocket
(env) $ pip install -e .
```

The web frontend must be served with HTTPS. Obtain a .pem file and save it as an environment variable named "LOCAL_CERT".

## Run
Simultaneously:
```
(env) $ python -m hololight.bci
```
```
(env) $ python frontend/serve.py
```

## TODO
Hardware
1. Assemble final hardware w/ head strap

Software  
1. Subprocess Train.py
1. philips hue unit for single light control
1. deploy to Pizero2

### BONUS
1. Dumb Web frontend for starting go-task and visualizing task
1. SSVEP Signal Processing chain
1. WebXR Task Frontend
