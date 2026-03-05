import numpy as np
from nn_engine import profiler

profiler.install()
profiler.enable()

from train_alexnet import train_alexnet
# Monkey patch to only run 1 epoch and 1 batch
import train_alexnet as ta
orig_train = ta.train_alexnet
def small_train():
    ta.epochs = 1
    # We will just let it run
    ta.train_alexnet()

profiler.disable()
profiler.uninstall()
small_train()