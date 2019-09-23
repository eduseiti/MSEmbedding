import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .datasets import HumanProteome

import numpy as np

import random

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import time


HOW_MANY_SAMPLES = 128


def main():

    dataset = HumanProteome.HumanProteome(
        "data/humanProteome",
        "eval",
        128,
        nb_threads = 1, 
        trainingDataset = None)

    batch_loader = dataset.make_batch_loader()

    for i, batch in enumerate(batch_loader):
        print(batch)


if __name__ == '__main__':
    main()

    