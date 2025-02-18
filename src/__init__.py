import sys
import os
import uuid
import math
from argparse import Namespace
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import open3d as o3d
import numpy as np
import cv2

from rich.console import Console

from tqdm import tqdm

import json

from einops import rearrange,repeat,reduce


from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
    Set,
)


# Config type
# from omegaconf import DictConfig

# PyTorch Tensor type
from torch import Tensor


# dataclasses
from dataclasses import dataclass, field