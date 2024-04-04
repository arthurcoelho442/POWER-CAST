# imports
import warnings
import pickle

import numpy as np
import pandas as pd

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks                                    import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers                                      import TensorBoardLogger

from pytorch_forecasting                                            import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data                                       import GroupNormalizer
from pytorch_forecasting.metrics                                    import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning  import optimize_hyperparameters

import seaborn as sns
import matplotlib.pyplot as plt

import tensorboard as tb
import tensorflow as tf 

# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile