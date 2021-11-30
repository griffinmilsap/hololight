from dataclasses import field
from pathlib import Path
import time

import ezmsg as ez

from ezmsg.builtins.websocket import WebsocketServer, WebsocketSettings

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
            

    @ez.subscriber( INPUT_LOG )
    @ez.publisher( OUTPUT_MODEL )
    async def train_model( self, message: Path ) -> AsyncGenerator:
        # Message is a path to a closed and complete training session recording
        model_out = self.SETTINGS.recording_dir / \
            self.STATE.training_session / \
            f'{self.STATE.time_str}.checkpoint'

        # TODO: Kickoff training script to crunch on all recordings in the session folder
        # TODO: Wait for training script to complete and write the checkpoint file

        # yield ( self.OUTPUT_MODEL, model_out )


class TrainingAPIState( ez.State ):
    ...

class TrainingAPI( ez.Unit ):

    STATE: TrainingAPIState

    INPUT_FROM_WEBSOCKET = ez.InputStream( ByteString )
    OUTPUT_TO_WEBSOCKET = ez.OutputStream( ByteString )

    OUTPUT_TRAINING = ez.OutputStream( ModelTrainingMessage )

    @ez.subscriber( INPUT_FROM_WEBSOCKET )
    async def on_input( self, input: ByteString ) -> None:
        ...


class ModelTrainingSettings( ez.Settings ):
    settings: ModelTrainingLogicSettings
    websocket_settings: WebsocketSettings
    logger_settings: MessageLoggerSettings = field(
        default_factory = MessageLoggerSettings
    )
    gotask_settings: GoTaskSettings = field(
        default_factory = GoTaskSettings
    )

class ModelTraining( ez.Collection ):
    SETTINGS: ModelTrainingSettings

    INPUT_LOGGER = ez.InputStream( ez.Message )
    OUTPUT_MODEL = ez.OutputStream( Path )

    GOTASK = GoTask()
    LOGGER = MessageLogger()
    LOGIC = ModelTrainingLogic()
    SERVER = WebsocketServer()
    API = TrainingAPI()

    def configure( self ) -> None:
        self.GOTASK.apply_settings( self.SETTINGS.gotask_settings )
        self.LOGGER.apply_settings( self.SETTINGS.logger_settings )
        self.LOGIC.apply_settings( self.SETTINGS.settings )
        self.SERVER.apply_settings( self.SETTINGS.websocket_settings )

    def network( self ) -> ez.NetworkDefinition:
        return (
            # Connections to Logic
            ( self.LOGIC.OUTPUT_TASK_START, self.GOTASK.INPUT_START ),
            ( self.GOTASK.OUTPUT_TASK, self.LOGIC.INPUT_TASK ),
            ( self.LOGGER.OUTPUT_STOP, self.LOGIC.INPUT_LOG ),
            ( self.LOGIC.OUTPUT_MODEL, self.OUTPUT_MODEL ),

            # Inputs to Data Logger
            ( self.LOGIC.OUTPUT_LOG_START, self.LOGGER.INPUT_START ),
            ( self.INPUT_LOGGER, self.LOGGER.INPUT_MESSAGE ),
            ( self.GOTASK.OUTPUT_TASK, self.LOGGER.INPUT_MESSAGE ),
            ( self.LOGIC.OUTPUT_LOG_STOP, self.LOGGER.INPUT_STOP ),

            # Training API
            ( self.SERVER.OUTPUT, self.API.INPUT_FROM_WEBSOCKET ),
            ( self.API.OUTPUT_TO_WEBSOCKET, self.SERVER.INPUT ),
            ( self.API.OUTPUT_TRAINING, self.LOGIC.INPUT_TRAIN )
        )