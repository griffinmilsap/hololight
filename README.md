# hololight

## Setup

```
$ python3 -m venv env
$ source env/bin/activate
(env) $ pip install --upgrade pip
(env) $ pip install -e ../path/to/ezmsg
(env) $ pip install -e ../path/to/ezmsg/extensions/ezmsg-sigproc
(env) $ pip install -e ../path/to/ezmsg/extensions/ezmsg-websocket
(env) $ pip install -e ../path/to/ezmsg/extensions/ezmsg-eeg
(env) $ pip install -e .
```

The web frontend must be served with HTTPS. Create a certificate with the following command

```bash
openssl req -new -x509 -days 365 -nodes -out cert.pem -keyout cert.pem
```

## Run

```
(env) $ python -m hololight.bci
```
