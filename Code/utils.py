import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler

def step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=10):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    return LearningRateScheduler(schedule)