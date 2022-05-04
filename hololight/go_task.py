import asyncio
import time
import random

from enum import Enum

import ezmsg.core as ez

from ezmsg.util.stampedmessage import StampedMessage
from ezmsg.testing.debuglog import DebugLog

from typing import (
    Any,
    AsyncGenerator,
    List,
    Optional
)

class GoTaskStage( str, Enum ):
    PRE_TASK = 'PRE-TASK'
    ACTIVITY = 'ACTIVITY'
    INTERTRIAL = 'INTERTRIAL'
    POST_TASK = 'POST-TASK'
    COMPLETE = 'COMPLETE'

class GoTaskMessage( StampedMessage ):
    stage: GoTaskStage

    # Only defined during stage.ACTIVITY
    trial_class: Optional[ int ] = None

class GoTaskSettings( ez.Settings ):
    pre_task_dur: float = 5.0 # sec
    activity_dur: float = 3.0 # sec
    intertrial_dur: float = 3.0 # sec
    post_task_dur: float = 5.0 # sec
    n_trials_per_class: int = 10

class GoTaskState( ez.State ):
    running: bool = False
    
class GoTask( ez.Unit ):
    SETTINGS: GoTaskSettings
    STATE: GoTaskState

    INPUT_START = ez.InputStream( int )
    OUTPUT_TASK = ez.OutputStream( GoTaskMessage )

    @ez.subscriber( INPUT_START )
    @ez.publisher( OUTPUT_TASK )
    async def go_task( self, n_classes: int ) -> AsyncGenerator:
        """ This task is triggered by publishing an integer to the input 
        The integer you publish is the desired number """

        if self.STATE.running:
            """ If we're already running, we just poop out here """
            return

        trials: List[ int ] = list( range( n_classes ) )
        trials *= self.SETTINGS.n_trials_per_class
        random.shuffle( trials )

        yield ( self.OUTPUT_TASK, GoTaskMessage( 
            stage = GoTaskStage.PRE_TASK
        ) )

        await asyncio.sleep( self.SETTINGS.pre_task_dur )

        for trial in trials:

            yield ( self.OUTPUT_TASK, GoTaskMessage(
                stage = GoTaskStage.ACTIVITY,
                trial_class = trial
            ) )

            await asyncio.sleep( self.SETTINGS.activity_dur )

            yield ( self.OUTPUT_TASK, GoTaskMessage(
                stage = GoTaskStage.INTERTRIAL
            ) )

            await asyncio.sleep( self.SETTINGS.intertrial_dur )

        yield ( self.OUTPUT_TASK, GoTaskMessage(
            stage = GoTaskStage.POST_TASK
        ) )

        await asyncio.sleep( self.SETTINGS.post_task_dur )

        yield ( self.OUTPUT_TASK, GoTaskMessage (
            stage = GoTaskStage.COMPLETE
        ) )

class _TaskKickoffSettings( ez.Settings ):
    n_classes: int = 2
    autostart_delay: Optional[ float ] = None

class _TaskKickoff( ez.Unit ):
    SETTINGS: _TaskKickoffSettings

    OUTPUT = ez.OutputStream( int )

    @ez.publisher( OUTPUT )
    async def kickoff( self ) -> AsyncGenerator:
        if self.SETTINGS.autostart_delay is not None:
            await asyncio.sleep( self.SETTINGS.autostart_delay ) 
        yield( self.OUTPUT, self.SETTINGS.n_classes )


class _GoTaskSystemSettings( ez.Settings ):
    gotask_settings: GoTaskSettings
    taskkickoff_settings: _TaskKickoffSettings

class _GoTaskSystem( ez.System ):

    SETTINGS: _GoTaskSystemSettings

    KICKOFF = _TaskKickoff()
    TASK = GoTask()
    PRINT = DebugLog()

    def configure( self ) -> None:
        self.TASK.apply_settings( self.SETTINGS.gotask_settings )
        self.KICKOFF.apply_settings( self.SETTINGS.taskkickoff_settings )

    def network( self ) -> ez.NetworkDefinition:
        return ( 
            ( self.KICKOFF.OUTPUT, self.TASK.INPUT_START ),
            ( self.TASK.OUTPUT_TASK, self.PRINT.INPUT ),
        )

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description = 'Go Task Test'
    )

    parser.add_argument(
        '--classes',
        type = int,
        help = 'Number of Classes',
        default = 1
    )

    parser.add_argument(
        '--pre-task',
        type = float,
        help = 'Number of seconds before trials start',
        default = 10.0
    )

    parser.add_argument(
        '--activity',
        type = float,
        help = 'Number of seconds for trial period',
        default = 3.0
    )

    parser.add_argument( 
        '--intertrial', 
        type = float, 
        help = 'Number of seconds for intertrial interval', 
        default = 3.0 
    )

    parser.add_argument(
        '--post-task',
        type = float,
        help = 'Number of seconds after the task',
        default = 10.0
    )

    parser.add_argument(
        '--trials',
        type = int,
        help = 'Number of trials per class; total trials = classes * trials',
        default = 3
    )

    args = parser.parse_args()

    settings = _GoTaskSystemSettings(
        gotask_settings = GoTaskSettings(
            pre_task_dur = args.pre_task, # sec
            activity_dur = args.activity, # sec
            intertrial_dur = args.intertrial, # sec
            post_task_dur = args.post_task, # sec
            n_trials_per_class = args.trials 
        ),
        taskkickoff_settings = _TaskKickoffSettings(
            n_classes = args.classes
        )
    )

    system = _GoTaskSystem( settings )

    ez.run_system( system )

    


