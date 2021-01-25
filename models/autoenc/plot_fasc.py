import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from pickle_eval import Evaluator
from config import *
from frechet import calculate_fid
from tqdm import tqdm
from generator import generator,vis_gen

test_generator = vis_gen(test_path)
model = load_model(model_dir)

# graph = []

def plotAboveThreshold(gen, threshold=3.3):
    k,n,filepath = next(gen)
    k,n = np.array(k), np.array(n)
    k = model.predict(k)
    summa = 0
    for i in range(30):
        loss = calculate_fid(k[0][0][i],n[i])
        summa +=loss
    summa /= 30
    if abs(summa) > threshold:
        evaluator = Evaluator(filepath, 10)
        evaluator.draw()

while True:
    plotAboveThreshold(test_generator)
