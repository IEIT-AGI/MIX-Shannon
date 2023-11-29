from enum import Enum


class RunningStage(str, Enum):
    FITTING = "fit"
    VALIDATING = "validate"
    TESTING = "test"
    PREDICTING = "predict"
    TUNING = "tune"
    TRAINING = "train"


def which_one_running_state(stage: Enum) -> str:
    _stage = stage.value
    return "train" if _stage is "fit" else _stage
