from typing import AsyncGenerator
import ezmsg.core as ez
import asyncio
import numpy as np

from ..hololight.shallowfbcspdecoder import DecoderOutput 
from ..hololight.hue import HueDemo, HueDemoSettings
from ..hololight.stamped_websocket_server import StampedTextMessage

class GenerateMessages( ez.Unit ):
  OUTPUT = ez.OutputStream(StampedTextMessage)

  @ez.publisher( OUTPUT )
  async def publish_messages( self ) -> AsyncGenerator:
    for i in range(0, 10000):
      await asyncio.sleep(.1)
      yield ( self.OUTPUT, StampedTextMessage(
        text = "1"
      ))

class GenerateDecode( ez.Unit ):
  OUTPUT = ez.OutputStream( DecoderOutput )

  @ez.publisher( OUTPUT )
  async def publish_decode( self ) -> AsyncGenerator:
    for i in range(0, 10000):
      await asyncio.sleep(.1)
      yield ( self.OUTPUT, DecoderOutput(
        output = np.array([0,1])
      ))

class HoloTestSystem( ez.System ):

  HUE = HueDemo()
  MESSAGES = GenerateMessages()
  DECODE = GenerateDecode()

  # def configure( self ) -> None:

  def network( self ) -> ez.NetworkDefinition:
    return (
      (self.MESSAGES.OUTPUT, self.HUE.INPUT_HOLOLENS),
      (self.DECODE.OUTPUT, self.HUE.INPUT_DECODE)
    )