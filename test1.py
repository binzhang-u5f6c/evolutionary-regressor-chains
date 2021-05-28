# -*- coding: utf-8 -*-
"""Test ERC with different rho.

Test ERC with different rho.
Modify streams in config.json to select different dataset.
The range of rho is range(0.1, 0.9, 0.1)
"""
from json import load
from copy import deepcopy

import torch

from erc.utils.loader import SyntheticStreams
from erc.model.baselearner import BaseLearnerSynthetic
from erc.model.chains import EvolutionaryRegressorChains

with open('config.json', 'r') as f:
    config = load(f)
device = torch.device(config['device'])
streams_name = 'synthetic'

print('Synthetic #1...')
print('loading data...')
streams = SyntheticStreams(1, device)
baselearner = BaseLearnerSynthetic
m = 8
print('Initialize models...')
erc = EvolutionaryRegressorChains(1, m, baselearner, device)
results = torch.zeros((50001, m+1), device=device)
prev_order = deepcopy(erc.chains[0].order)
for i, data in enumerate(streams):
    if i % 100 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    results[i, :-1] = erc.step(x, y)
    if torch.equal(prev_order, erc.chains[0].order):
        results[i+1, -1] = results[i, -1]
    else:
        prev_order = deepcopy(erc.chains[0].order)
        results[i+1, -1] = results[i, -1] + 1
torch.save(results.detach(),
           'results/{}1.pt'.format(streams_name))

print('Synthetic #2...')
print('loading data...')
streams = SyntheticStreams(2, device)
baselearner = BaseLearnerSynthetic
m = 7
print('Initialize models...')
erc = EvolutionaryRegressorChains(1, m, baselearner, device)
results = torch.zeros((50001, m+1), device=device)
prev_order = deepcopy(erc.chains[0].order)
for i, data in enumerate(streams):
    if i % 100 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    results[i, :-1] = erc.step(x, y)
    if torch.equal(prev_order, erc.chains[0].order):
        results[i+1, -1] = results[i, -1]
    else:
        prev_order = deepcopy(erc.chains[0].order)
        results[i+1, -1] = results[i, -1] + 1
torch.save(results.detach(),
           'results/{}2.pt'.format(streams_name))

print('Synthetic #3...')
print('loading data...')
streams = SyntheticStreams(3, device)
baselearner = BaseLearnerSynthetic
m = 9
print('Initialize models...')
erc = EvolutionaryRegressorChains(1, m, baselearner, device)
results = torch.zeros((50001, m+1), device=device)
prev_order = deepcopy(erc.chains[0].order)
for i, data in enumerate(streams):
    if i % 100 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    results[i, :-1] = erc.step(x, y)
    if torch.equal(prev_order, erc.chains[0].order):
        results[i+1, -1] = results[i, -1]
    else:
        prev_order = deepcopy(erc.chains[0].order)
        results[i+1, -1] = results[i, -1] + 1
torch.save(results.detach(),
           'results/{}3.pt'.format(streams_name))

print('Synthetic #4...')
print('loading data...')
streams = SyntheticStreams(4, device)
baselearner = BaseLearnerSynthetic
m = 6
print('Initialize models...')
erc = EvolutionaryRegressorChains(1, m, baselearner, device)
results = torch.zeros((50001, m+1), device=device)
prev_order = deepcopy(erc.chains[0].order)
for i, data in enumerate(streams):
    if i % 100 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    results[i, :-1] = erc.step(x, y)
    if torch.equal(prev_order, erc.chains[0].order):
        results[i+1, -1] = results[i, -1]
    else:
        prev_order = deepcopy(erc.chains[0].order)
        results[i+1, -1] = results[i, -1] + 1
torch.save(results.detach(),
           'results/{}4.pt'.format(streams_name))

print('Synthetic #5...')
print('loading data...')
streams = SyntheticStreams(5, device)
baselearner = BaseLearnerSynthetic
m = 8
print('Initialize models...')
erc = EvolutionaryRegressorChains(1, m, baselearner, device)
results = torch.zeros((50001, m+1), device=device)
prev_order = deepcopy(erc.chains[0].order)
for i, data in enumerate(streams):
    if i % 100 == 0:
        print('  Processing sample {}...'.format(i+1))
    x, y = data
    results[i, :-1] = erc.step(x, y)
    if torch.equal(prev_order, erc.chains[0].order):
        results[i+1, -1] = results[i, -1]
    else:
        prev_order = deepcopy(erc.chains[0].order)
        results[i+1, -1] = results[i, -1] + 1
torch.save(results.detach(),
           'results/{}5.pt'.format(streams_name))
