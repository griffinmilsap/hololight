import ezmsg as ez
import numpy as np

from phue import Bridge

from .shallowfbcspdecoder import DecoderOutput

from typing import (
    Optional
)

class HueDemoSettings( ez.Settings ):
    trigger_class: int = 1
    trigger_thresh: float = 0.5
    bridge_host: Optional[ str ] = None

class HueDemoState( ez.State ):
    lights_on: bool = False
    decode_class: Optional[ int ] = None
    bridge: Optional[ Bridge ] = None

class HueDemo( ez.Unit ):
    SETTINGS: HueDemoSettings
    STATE: HueDemoState

    INPUT_DECODE = ez.InputStream( DecoderOutput )

    def initialize( self ) -> None:
        if self.SETTINGS.bridge_host:
            # Connect to bridge
            # Set lights to initial state
            self.STATE.bridge = Bridge( self.SETTINGS.bridge_host )
            self.STATE.bridge.connect()

            self.set_lights( self.STATE.lights_on )

    def set_lights( self, on: bool ):
        print( 'Set lights', 'on' if on else 'off' )
        if self.STATE.bridge:
            for light in self.STATE.bridge.lights:
                if light.reachable:
                    light.on = on

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
                    self.STATE.lights_on = not self.STATE.lights_on
                    self.set_lights( self.STATE.lights_on )








