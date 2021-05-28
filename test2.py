# -*- coding: utf-8 -*-
"""Test ERC with different n_chains.

Test ERC with different n_chains.
Modify streams in config.json to select different dataset.
ERC with 1/10/20/50/100 chains are evaluated.
"""
from time import time
from json import load

import torch

from erc.utils.loader import WeatherStreams
from erc.utils.loader import TrainStreams
from erc.utils.loader import SensorStreams
from erc.model.baselearner import BaseLearnerWeather
from erc.model.baselearner import BaseLearnerTrain
from erc.model.baselearner import BaseLearnerSensor
from erc.model.chains import EvolutionaryRegressorChains

with open('config.json', 'r') as f:
    config = load(f)
device = torch.device(config['device'])
streams_name = config["streams"]

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
erc = EvolutionaryRegressorChains(1, m, baselearner, device)
results = torch.zeros(m+1, device=device)
t1 = time()
for i, data in enumerate(streams):
    if i % 100 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    results[:-1] += erc.step(x, y)
t2 = time()
results[m] = t2 - t1
torch.save(results.detach(),
           'results/{}1.pt'.format(streams_name))

print('Initialize models...')
erc = EvolutionaryRegressorChains(5, m, baselearner, device)
results = torch.zeros(m+1, device=device)
t1 = time()
for i, data in enumerate(streams):
    if i % 100 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    results[:-1] += erc.step(x, y)
t2 = time()
results[m] = t2 - t1
torch.save(results.detach(),
           'results/{}2.pt'.format(streams_name))

print('Initialize models...')
erc = EvolutionaryRegressorChains(10, m, baselearner, device)
results = torch.zeros(m+1, device=device)
t1 = time()
for i, data in enumerate(streams):
    if i % 100 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    results[:-1] += erc.step(x, y)
t2 = time()
results[m] = t2 - t1
torch.save(results.detach(),
           'results/{}3.pt'.format(streams_name))

print('Initialize models...')
erc = EvolutionaryRegressorChains(20, m, baselearner, device)
results = torch.zeros(m+1, device=device)
t1 = time()
for i, data in enumerate(streams):
    if i % 100 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    results[:-1] += erc.step(x, y)
t2 = time()
results[m] = t2 - t1
torch.save(results.detach(),
           'results/{}4.pt'.format(streams_name))

print('Initialize models...')
erc = EvolutionaryRegressorChains(50, m, baselearner, device)
results = torch.zeros(m+1, device=device)
t1 = time()
for i, data in enumerate(streams):
    if i % 100 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    results[:-1] += erc.step(x, y)
t2 = time()
results[m] = t2 - t1
torch.save(results.detach(), 'results/train5.pt')

print('Initialize models...')
erc = EvolutionaryRegressorChains(100, m, baselearner, device)
results = torch.zeros(m+1, device=device)
t1 = time()
for i, data in enumerate(streams):
    if i % 100 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    results[:-1] += erc.step(x, y)
t2 = time()
results[m] = t2 - t1
torch.save(results.detach(),
           'results/{}5.pt'.format(streams_name))
