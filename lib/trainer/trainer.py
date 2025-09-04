from lib.trainer.base import *
from lib.trainer.integer import ContinuousCorrExtensionIntegerTrainer

available_trainers = [
    ContrastiveLossTrainer,
    HardestContrastiveLossTrainer,
    TripletLossTrainer,
    HardestTripletLossTrainer,
    CorrespondenceExtensionTrainer,
    ContinuousCorrExtensionTrainer,
    ContinuousHardestContrastiveTrainer,
    ContinuousCorrExtensionIntegerTrainer,
]

__all_trainers__ = {t.__name__: t for t in available_trainers}

def get_trainer(trainer):
    if trainer in __all_trainers__:
        return __all_trainers__[trainer]
    else:
        raise ValueError(f"Trainer {trainer} not found")
