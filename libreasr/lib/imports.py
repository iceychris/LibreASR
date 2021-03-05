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

# fastai
from fastai.torch_basics import *
from fastai.layers import *
from fastai.data.all import *
from fastai.optimizer import *
from fastai.learner import *
from fastai.metrics import *
from fastai.text.core import *
from fastai.text.data import *
from fastai.text.models.core import *
from fastai.text.models.awdlstm import *
from fastai.text.learner import *
from fastai.callback.rnn import *
from fastai.callback.all import *

from fastai.distributed import rank0_first
from fastai.torch_core import rank_distrib

from fastai.vision.learner import *
from fastai.vision.models.xresnet import *

# other
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

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

# local audio
from fastaudio.core import *
from fastaudio.augment import *
