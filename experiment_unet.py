from config import *
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device_ids = [0]
import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from datetime import datetime
from time import time
import torch.nn.functional as F
import json
import random
import pickle
from argparse import ArgumentParser
from data.dataset import FullSizeVideoDataset
from models.unet3D import Simply3DUnet
# models.unet.U Net_alternative ALTERNATIVE HAS SIGMOID ACTIVATION!!!
import numpy as np


# model_used = resnet10(num_classes=5, activation="softmax")
# model_used = MyModel


class Run():
    def __init__(self, batch_size, run_name="run_run"):
        self.run_dir = os.path.join("./files", run_name)
        self.batch_size = batch_size
        self.model = Simply3DUnet(num_in_channels=1, num_out_channels=1, depth=1)
        self.model = self.model.to(device)
        self.dataset = FullSizeVideoDataset
        self.data_train = DataLoader(self.dataset("./files/train"), batch_size=self.batch_size)
        self.data_val = DataLoader(self.dataset("./files/val"), batch_size=self.batch_size)
        self.data_test = DataLoader(self.dataset("./files/test"), batch_size=self.batch_size)

        self.loss = torch.nn.MSELoss(reduction='mean')
        self.params = self.model.parameters()
        self.optimizer = torch.optim.AdamW(params=self.params, lr=3e-4)
        self.summary = SummaryWriter(log_dir=os.path.join(self.run_dir, "summary"))
        self.global_step = 0
        self.train_step = 0
        self.val_step = 0

    def forward(self, input):
        input = input.to(device)
        inp = input[:, :, :10].float()
        target = input[:, :, 10:].float()
        inp.to(device)
        otpt = self.model(inp)
        loss = self.loss(otpt, target=target)
        return loss, otpt

    def epoch_train(self):
        self.model.train()
        epoch_loss = 0
        for data in tqdm(self.data_train, desc="training"):
            if random.randint(0, 10) != 5:
                continue
            self.optimizer.zero_grad()
            loss, otpt = self.forward(data)
            loss.backward()
            self.optimizer.step()
            loss = loss.detach().cpu().numpy()
            self.summary.add_scalar("batch_train_loss", loss, global_step=self.train_step)
            epoch_loss += loss
            self.train_step += 1
        epoch_loss /= len(self.data_train)
        return epoch_loss

    def epoch_val(self):
        self.model.eval()
        epoch_loss = 0
        for data in tqdm(self.data_val, desc="validating"):
            if random.randint(0, 10) != 5:  # sam 1 na 10 izvajaj...
                continue
            with torch.no_grad():
                loss, otpt = self.forward(data)
                loss = loss.detach().cpu().numpy()
                self.summary.add_scalar("batch_val_loss", loss, global_step=self.val_step)
                epoch_loss += loss
        epoch_loss /= len(self.data_val)
        return epoch_loss

    def train(self, no_epochs=10):
        best_train_loss = 1e9
        best_val_loss = 1e9
        for i in range(no_epochs):
            tr = self.epoch_train()
            val = self.epoch_val()
            self.summary.add_scalars(main_tag="losses", tag_scalar_dict={"train_loss": tr, "val_loss": val},
                                     global_step=i)
            self.global_step += 1

            if val < best_val_loss:
                with open(os.path.join(self.run_dir, "best_val.pth"), "wb") as f:
                    torch.save(self.model, f)
                    print("best val saved")
                    best_val_loss = val
            print(f"STEP: {self.global_step} TRAINLOSS: {tr} VALLOSS {val}")
        self.summary.close()


from pprint import pprint

if __name__ == "__main__":
    r = Run(2, run_name="run_run_load")
    r.model = torch.load("/media/leon/2tbssd/ULTRAZVOK_COLLAB/ULTRASOUND_MASTER/files/run_run/best_val.pth")
    r.data_val = r.data_test
    print(r.epoch_val())
