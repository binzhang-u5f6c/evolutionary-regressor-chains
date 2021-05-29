# -*- coding: utf-8 -*-
"""Test ERC with different rho.

Test ERC with different rho.
Modify streams in config.json to select different dataset.
ERC with rho = range(0.1, 0.9, 0.1) are evaluated.
"""
from time import time
from json import load, dump

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
           'results/{}.1.pt'.format(streams_name))
config["rho"] = 0.2
with open('config.json', 'w') as f:
    dump(config, f)

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
           'results/{}.2.pt'.format(streams_name))
config["rho"] = 0.3
with open('config.json', 'w') as f:
    dump(config, f)

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
           'results/{}.3.pt'.format(streams_name))
config["rho"] = 0.4
with open('config.json', 'w') as f:
    dump(config, f)

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
           'results/{}.4.pt'.format(streams_name))
config["rho"] = 0.5
with open('config.json', 'w') as f:
    dump(config, f)

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
           'results/{}.5.pt'.format(streams_name))
config["rho"] = 0.6
with open('config.json', 'w') as f:
    dump(config, f)

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
           'results/{}.6.pt'.format(streams_name))
config["rho"] = 0.7
with open('config.json', 'w') as f:
    dump(config, f)

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
           'results/{}.7.pt'.format(streams_name))
config["rho"] = 0.8
with open('config.json', 'w') as f:
    dump(config, f)

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
           'results/{}.8.pt'.format(streams_name))
config["rho"] = 0.9
with open('config.json', 'w') as f:
    dump(config, f)

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
           'results/{}.9.pt'.format(streams_name))
