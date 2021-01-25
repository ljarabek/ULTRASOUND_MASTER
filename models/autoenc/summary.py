#!/usr/bin/env python

import pickle
import os
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import logging
import argparse
import tensorflow

from generator import generator
from timeit import default_timer as timer
from tensorflow.keras.models import load_model
from autoencoder import plot,history

# PLOT TRAINING HISTORY

plot(history)




