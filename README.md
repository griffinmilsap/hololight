# hololight

## Setup

```
$ python3 -m venv env
$ source env/bin/activate
(env) $ pip install --upgrade pip
(env) $ pip install ../path/to/ezmsg
(env) $ pip install ../path/to/ezbci
(env) $ pip install -r requirements.txt
```

## Run
```
(env) $ python -m hololight.bci
```

## TODO
Hardware
1. Make 4x electrode connectors
1. Assemble final hardware w/ head strap

Software
1. Triggerable Go task w/ dynamic recording
1. Modify Shallow FPCSP to dynamically load in new models
1. Subprocess Train.py
1. philips hue unit for single light control
1. deploy to Pizero2

### BONUS
1. Dumb Web frontend for starting go-task and visualizing task
1. SSVEP Signal Processing chain
1. WebXR Task Frontend