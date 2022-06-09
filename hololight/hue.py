import ezmsg.core as ez
import numpy as np
import time

from phue import Bridge

from .shallowfbcspdecoder import DecoderOutput
from .stamped_websocket_server import StampedTextMessage

from typing import (
    Optional,
    List
)

ON = True
OFF = False

class HueDemoSettings( ez.Settings ):
    trigger_class: int = 1
    trigger_thresh: float = 0.9
    bridge_host: Optional[ str ] = None
    num_lights: Optional[ int ] = None

# Turn lights_on into a vector of bools
class HueDemoState( ez.State ):
    lights_on: List[ bool ] = None
    decode_class: Optional[ int ] = None
    bridge: Optional[ Bridge ] = None
    hololens_state: Optional[ StampedTextMessage ] = None

class HueDemo( ez.Unit ):
    SETTINGS: HueDemoSettings
    STATE: HueDemoState

    INPUT_DECODE = ez.InputStream( DecoderOutput )
    INPUT_HOLOLENS = ez.InputStream( StampedTextMessage )

    def initialize( self ) -> None:
        if self.SETTINGS.bridge_host:
            # Connect to bridge
            # Set lights to initial state
            self.STATE.bridge = Bridge( self.SETTINGS.bridge_host )
            self.STATE.bridge.connect()
            if not self.SETTINGS.num_lights:
                self.SETTINGS.num_lights = len(self.STATE.bridge.lights)
            self.STATE.lights_on = [OFF for i in range(self.SETTINGS.num_lights)]
            self.set_all_lights( OFF )

        else:
            print("No bridge detected.", self.SETTINGS.num_lights, "lights will be simulated.")
            self.STATE.lights_on = [ OFF for i in range(self.SETTINGS.num_lights) ]

    def set_all_lights( self, on: bool ):
        for i, light in enumerate(self.STATE.lights_on):
            self.set_light(i, on)

    def set_light( self, i: int, on: bool ):
        print( 'Set light', i, 'on' if on else 'off' )
        self.STATE.lights_on[i] = on
        if self.STATE.bridge:
            light = self.STATE.bridge.lights[i]
            if light.reachable:
                light.on = on
    
    def flip_light( self, i: int ):
        current_state = self.STATE.lights_on[i]
        print( 'Set light', i, 'off' if current_state else 'on' )
        self.STATE.lights_on[i] = not self.STATE.lights_on[i]
        if self.STATE.bridge:
            light = self.STATE.bridge.lights[i]
            if light.reachable:
                light.on = not light.on


    # Post most recently received message to state
    @ez.subscriber( INPUT_HOLOLENS )
    async def on_hololens_input( self, hololens_message: StampedTextMessage ) -> None:
        self.STATE.hololens_state = hololens_message

    # If timestamp on ws message in state is older than a half second, don't use it
    # If timestamp is recent enough, turn light on
    @ez.subscriber( INPUT_DECODE )
    async def on_decode( self, decode: DecoderOutput ) -> None:
        probs = np.exp( decode.output )
        cur_class = probs.argmax()
        cur_prob = probs[ cur_class ]

        print( cur_class, cur_prob )

        if self.STATE.decode_class is None:
            self.STATE.decode_class = cur_class

        if cur_class != self.STATE.decode_class:
            if cur_prob > self.SETTINGS.trigger_thresh:
                self.STATE.decode_class = cur_class

                if cur_class == self.SETTINGS.trigger_class:

                    ws_message = self.STATE.hololens_state
                    ws_message_age = time.time() - ws_message._timestamp
                    if ws_message_age < 0.5:
                        self.flip_light(int(ws_message.text))

    @ez.main
    def dummy( self ):
        pass





