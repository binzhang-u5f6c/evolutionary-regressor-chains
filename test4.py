# -*- coding: utf-8 -*-
"""Evaluate ERC compared with some methods.

Evaluate ERC compared with some methods.
* Single models
* One-chain ERC
* Vanila RC
* Online RC
"""
from time import time
from json import load

import torch
import torch.nn.functional as F

from erc.utils.loader import WeatherStreams
from erc.utils.loader import TrainStreams
from erc.utils.loader import SensorStreams
from erc.model.baselearner import BaseLearnerWeather
from erc.model.baselearner import BaseLearnerTrain
from erc.model.baselearner import BaseLearnerSensor
from erc.model.others import SingleModels
from erc.model.others import RegressorChains
from erc.model.chains import EvolutionaryRegressorChains

with open('config.json', 'r') as f:
    config = load(f)
device = torch.device(config['device'])
streams_name = config["streams"]
n_chains = config["n_chains"]
n_pruning = config["n_pruning"]

print('loading data...')
if streams_name == "weather":
    streams = WeatherStreams(device)
    baselearner = BaseLearnerWeather
    m = 10
elif streams_name == "train":
    streams = TrainStreams(device)
    baselearner = BaseLearnerTrain
    m = 8
elif streams_name == "sensor":
    streams = SensorStreams(device)
    baselearner = BaseLearnerSensor
    m = 6

print('Initialize models...')
sm = SingleModels(m, baselearner, device)
results = torch.zeros(m+1, device=device)
t1 = time()
for i, data in enumerate(streams):
    if i % 1000 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    results[:-1] += sm.step(x, y)
t2 = time()
results[m] = t2 - t1
torch.save(results.detach(), 'results/single_model.pt')

print('Initialize models...')
rc = EvolutionaryRegressorChains(1, m, baselearner, device)
results = torch.zeros(m+1, device=device)
t1 = time()
for i, data in enumerate(streams):
    if i % 1000 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    results[:-1] += rc.step(x, y)
t2 = time()
results[m] = t2 - t1
torch.save(results.detach(), 'results/one_chain_erc.pt')

print('Initialize models...')
vrc = RegressorChains(n_chains, m, baselearner, device)
results = torch.zeros(m+1, device=device)
t1 = time()
for i, data in enumerate(streams):
    if i % 1000 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    if i < 2000:
        results[:-1] += vrc.step(x, y)
    else:
        y_hat = vrc.predict(x)
        results[:-1] += F.mse_loss(y_hat, y, reduction='none')
t2 = time()
results[m] = t2 - t1
torch.save(results.detach(), 'results/vanilarc.pt')

print('Initialize models...')
orc = RegressorChains(n_chains, m, baselearner, device)
results = torch.zeros(m+1, device=device)
t1 = time()
for i, data in enumerate(streams):
    if i % 100 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    results[:-1] += orc.step(x, y)
t2 = time()
results[m] = t2 - t1
torch.save(results.detach(), 'results/onlinerc.pt')

print('Initialize models...')
erc = EvolutionaryRegressorChains(n_chains, m, baselearner, device)
results = torch.zeros(m+1, device=device)
t1 = time()
for i, data in enumerate(streams):
    if i % 1000 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    results[:-1] += erc.step(x, y, n_pruning=n_pruning)
t2 = time()
results[m] = t2 - t1
torch.save(results.detach(), 'results/erc.pt')
