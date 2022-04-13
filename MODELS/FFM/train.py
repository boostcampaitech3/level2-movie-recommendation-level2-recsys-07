import argparse, yaml
import glob
from importlib import import_module
import multiprocessing
import os
import random
import re
import csv
import pandas as pd

from tqdm.auto import tqdm

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, Dataset

import mlflow

from model import *

#seed fix
def seed_setting(seed):

    # cpu seed
    torch.manual_seed(seed)

    # GPU seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you use multi GPU

    # CuDDN option
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # numpy rand seed
    np.random.seed(seed)

    # random seed
    random.seed(seed)

