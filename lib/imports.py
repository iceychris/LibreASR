# add flac to supported audio types
import mimetypes

mimetypes.types_map[".flac"] = "audio/flac"

# stock
import os
from functools import partial
import multiprocessing
import math
import sys

# fix env vars
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_SILENT"] = "True"

# IPython
from IPython.core.debugger import set_trace
from IPython.display import Audio, display

# fastai2
from fastai2.torch_basics import *
from fastai2.layers import *
from fastai2.data.all import *
from fastai2.optimizer import *
from fastai2.learner import *
from fastai2.metrics import *
from fastai2.text.core import *
from fastai2.text.data import *
from fastai2.text.models.core import *
from fastai2.text.models.awdlstm import *
from fastai2.text.learner import *
from fastai2.callback.rnn import *
from fastai2.callback.all import *

from fastai2.vision.learner import *
from fastai2.vision.models.xresnet import *

from fastai2_audio.core import *
from fastai2_audio.augment import *

# other
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import librosa
import librosa.display

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 175

# local
from lib.config import parse_and_apply_config
from lib.models import *
from lib.utils import *
from lib.callbacks import *
from lib.transforms import *
from lib.language import get_language
from lib.loss import *
from lib.metrics import *
from lib.data import *
from lib.decoders import *
from lib.builder import ASRDatabunchBuilder

# apply patches
import lib.patches
