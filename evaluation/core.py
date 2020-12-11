from config import *
from data.video_data_generation import array_from_video
import torch
from models.unet3D import Simply3DUnet
import random as rand
import numpy as np
import os
import pickle
from data.dataset import FullSizeVideoDataset
from tqdm import tqdm
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
