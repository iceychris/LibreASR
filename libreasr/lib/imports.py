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
from libreasr.lib.config import parse_and_apply_config
from libreasr.lib.models import *
from libreasr.lib.utils import *
from libreasr.lib.callbacks import *
from libreasr.lib.transforms import *
from libreasr.lib.language import get_language
from libreasr.lib.loss import *
from libreasr.lib.metrics import *
from libreasr.lib.data import *
from libreasr.lib.decoders import *
from libreasr.lib.builder import ASRDatabunchBuilder

# apply patches
import libreasr.lib.patches
