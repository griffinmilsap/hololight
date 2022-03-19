import asyncio

from dataclasses import field, replace
from pathlib import Path
import time

import ezmsg as ez

from ezmsg.builtins.websocket import WebsocketServer, WebsocketSettings
from ezmsg.ezbci.eegmessage import EEGInfoMessage, EEGMessage, EEGDataMessage

from .go_task import GoTask, GoTaskMessage, GoTaskSettings, GoTaskStage
from .messagelogger import MessageLogger, MessageLoggerSettings

from typing import (
    AsyncGenerator,
    ByteString,
    Optional
)

class ModelTrainingMessage( ez.Message ):
    n_classes: int
    session: str

class ModelTrainingLogicSettings( ez.Settings ):
    recording_dir: Path

class ModelTrainingLogicState( ez.State ):
    training_session: Optional[ ModelTrainingMessage ] = None
    data_log: Optional[ Path ] = None
    time_str: Optional[ str ] = None

class ModelTrainingLogic( ez.Unit ):
    """ Coordinates model training cycle """

    SETTINGS: ModelTrainingLogicSettings
    STATE: ModelTrainingLogicState
    
    INPUT_TRAIN = ez.InputStream( ModelTrainingMessage )
    INPUT_TASK = ez.InputStream( GoTaskMessage )
    OUTPUT_TASK_START = ez.OutputStream( int )
    OUTPUT_LOG_START = ez.OutputStream( Path )
    OUTPUT_LOG_STOP = ez.OutputStream( Path )
    INPUT_LOG = ez.InputStream( Path )
    OUTPUT_MODEL = ez.OutputStream( Path )

    @ez.subscriber( INPUT_TRAIN )
    @ez.publisher( OUTPUT_TASK_START )
    @ez.publisher( OUTPUT_LOG_START )
    async def kickoff_task( self, message: ModelTrainingMessage ) -> AsyncGenerator:
        if self.STATE.training_session is not None:
            # TODO: Issue warning
            return

        self.STATE.training_session = message

        self.STATE.time_str = time.strftime( '%Y%m%dT%H%M%S' )
        self.STATE.data_log = self.SETTINGS.recording_dir / \
            self.STATE.training_session.session / \
            f'{self.STATE.time_str}.txt'

        # Kickoff logging and the task
        yield ( self.OUTPUT_LOG_START, self.STATE.data_log )
        yield ( self.OUTPUT_TASK_START, self.STATE.training_session.n_classes )

    @ez.subscriber( INPUT_TASK )
    @ez.publisher( OUTPUT_LOG_STOP )
    async def monitor_task( self, message: GoTaskMessage ) -> AsyncGenerator:
        if message.stage == GoTaskStage.COMPLETE:
            yield ( self.OUTPUT_LOG_STOP, self.STATE.data_log )
            print( 'Training Complete' )
            

    @ez.subscriber( INPUT_LOG )
    @ez.publisher( OUTPUT_MODEL )
    async def train_model( self, message: Path ) -> AsyncGenerator:
        # Message is a path to a closed and complete training session recording
        train_dir = self.SETTINGS.recording_dir / \
            self.STATE.training_session.session
        tag = f'{self.STATE.time_str}'
        
        print( 'Kicking off Train script...' )

        proc = await asyncio.create_subprocess_shell(
            f'python -m hololight.train --dir={str(train_dir)} --tag={tag}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await proc.communicate()

        print(f'[Train script exited with {proc.returncode}]')

        if stdout:
            print(f'[stdout]\n{stdout.decode()}')
        if stderr:
            print(f'[stderr]\n{stderr.decode()}')

        yield ( self.OUTPUT_MODEL, train_dir / f'{tag}.checkpoint' )

        self.STATE.training_session = None

import numpy as np

class TestSignalInjectorSettings( ez.Settings ):
    enabled: bool = False

class TestSignalInjectorState( ez.State ):
    info: Optional[ EEGInfoMessage ] = None
    task_state: Optional[ GoTaskMessage ] = None
    cur_sample: int = 0

class TestSignalInjector( ez.Unit ):
    """ Injects a test signal during specific task stages """
    SETTINGS: TestSignalInjectorSettings
    STATE: TestSignalInjectorState

    INPUT_TASK = ez.InputStream( GoTaskMessage )
    INPUT_SIGNAL = ez.InputStream( EEGMessage )
    OUTPUT_SIGNAL = ez.OutputStream( EEGMessage )

    @ez.subscriber( INPUT_TASK )
    async def on_task_update( self, message: GoTaskMessage ) -> None:
        self.STATE.task_state = message
        if message.stage == GoTaskStage.COMPLETE:
            self.STATE.task_state = None

    @ez.subscriber( INPUT_SIGNAL )
    @ez.publisher( OUTPUT_SIGNAL )
    async def on_signal( self, message: EEGMessage ) -> AsyncGenerator:

        # If the module isn't enabled, we just passthrough and republish
        if not self.SETTINGS.enabled:
            yield ( self.OUTPUT_SIGNAL, message )
            return

        if isinstance( message, EEGInfoMessage ):
            self.STATE.info = message

        elif isinstance( message, EEGDataMessage ):
            if self.STATE.info is not None:
                if self.STATE.task_state is not None:

                    # On intertrial periods, we reset test signal phase
                    if self.STATE.task_state.stage == GoTaskStage.INTERTRIAL:
                        self.STATE.cur_sample = 0

                    # On activity periods, we add a test signal to the data stream
                    elif self.STATE.task_state.stage == GoTaskStage.ACTIVITY:
                        freq = 10.0 + ( 5.0 * self.STATE.task_state.trial_class )
                        t = np.arange( self.STATE.info.n_time ) + self.STATE.cur_sample
                        self.STATE.cur_sample += self.STATE.info.n_time
                        signal = np.sin( 2.0 * np.pi * freq * ( t / self.STATE.info.fs ) )
                        message = replace( message, data = ( message.data.T + signal ).T ) # broadcasting
                        print( f'Injecting {freq} hz signal for class {self.STATE.task_state.trial_class}...' )

        yield ( self.OUTPUT_SIGNAL, message )

import json

class TrainingAPIState( ez.State ):
    ...

class TrainingAPI( ez.Unit ):

    STATE: TrainingAPIState

    INPUT_FROM_WEBSOCKET = ez.InputStream( bytes )
    OUTPUT_TO_WEBSOCKET = ez.OutputStream( bytes )

    INPUT_TASK = ez.InputStream( GoTaskMessage )
    OUTPUT_TRAINING = ez.OutputStream( ModelTrainingMessage )

    @ez.subscriber( INPUT_FROM_WEBSOCKET )
    @ez.publisher( OUTPUT_TRAINING )
    async def on_input( self, input: bytes ) -> None:
        # message = json.loads( input.decode( 'utf-8' ) )

        # print( message )
        print( 'Message Received' )

        yield( self.OUTPUT_TRAINING, 
            ModelTrainingMessage(
                n_classes = 2, 
                session = 'TEST' 
            ) 
        )

    @ez.subscriber( INPUT_TASK )
    async def on_task_stage( self, task_msg: GoTaskMessage ):
        print( task_msg )



class ModelTrainingSettings( ez.Settings ):
    settings: ModelTrainingLogicSettings
    websocket_settings: WebsocketSettings
    testsignal_settings: TestSignalInjectorSettings = field(
        default_factory = TestSignalInjectorSettings
    )
    logger_settings: MessageLoggerSettings = field(
        default_factory = MessageLoggerSettings
    )
    gotask_settings: GoTaskSettings = field(
        default_factory = GoTaskSettings
    )

class ModelTraining( ez.Collection ):
    SETTINGS: ModelTrainingSettings

    INPUT_LOGGER = ez.InputStream( ez.Message )
    INPUT_SIGNAL = ez.InputStream( EEGMessage )
    OUTPUT_MODEL = ez.OutputStream( Path )

    GOTASK = GoTask()
    TEST = TestSignalInjector()
    LOGGER = MessageLogger()
    LOGIC = ModelTrainingLogic()
    SERVER = WebsocketServer()
    API = TrainingAPI()

    def configure( self ) -> None:
        self.GOTASK.apply_settings( self.SETTINGS.gotask_settings )
        self.LOGGER.apply_settings( self.SETTINGS.logger_settings )
        self.LOGIC.apply_settings( self.SETTINGS.settings )
        self.SERVER.apply_settings( self.SETTINGS.websocket_settings )
        self.TEST.apply_settings( self.SETTINGS.testsignal_settings )

    def network( self ) -> ez.NetworkDefinition:
        return (
            # Connections to Logic
            ( self.LOGIC.OUTPUT_TASK_START, self.GOTASK.INPUT_START ),
            ( self.GOTASK.OUTPUT_TASK, self.LOGIC.INPUT_TASK ),
            ( self.LOGGER.OUTPUT_STOP, self.LOGIC.INPUT_LOG ),
            ( self.LOGIC.OUTPUT_MODEL, self.OUTPUT_MODEL ),

            # Test Signal Injector
            ( self.GOTASK.OUTPUT_TASK, self.TEST.INPUT_TASK ),
            ( self.INPUT_SIGNAL, self.TEST.INPUT_SIGNAL ),

            # Inputs to Data Logger
            ( self.TEST.OUTPUT_SIGNAL, self.LOGGER.INPUT_SIGNAL ),
            ( self.LOGIC.OUTPUT_LOG_START, self.LOGGER.INPUT_START ),
            ( self.INPUT_LOGGER, self.LOGGER.INPUT_MESSAGE ),
            ( self.GOTASK.OUTPUT_TASK, self.LOGGER.INPUT_MESSAGE ),
            ( self.LOGIC.OUTPUT_LOG_STOP, self.LOGGER.INPUT_STOP ),

            # Training API
            ( self.SERVER.OUTPUT, self.API.INPUT_FROM_WEBSOCKET ),
            ( self.API.OUTPUT_TO_WEBSOCKET, self.SERVER.INPUT ),
            ( self.API.OUTPUT_TRAINING, self.LOGIC.INPUT_TRAIN ),
            ( self.GOTASK.OUTPUT_TASK, self.API.INPUT_TASK )
        )