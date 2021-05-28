# -*- coding: utf-8 -*-
"""Baseline models.

Models of baselines
"""
import torch
import torch.nn.functional as F

from ..utils.loader import TrainStreams
from ..utils.loader import WeatherStreams
from ..utils.loader import SensorStreams
from .baselearner import BaseLearnerTrain
from .baselearner import BaseLearnerWeather
from .baselearner import BaseLearnerSensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SingleModels:
    """Single models.

    Build single model for every data stream.
    """

    def __init__(self, L, baselearner, device=device):
        """__init__ for SingleModels."""
        super(SingleModels, self).__init__()
        self.learners = [baselearner(device=device)
                         for _ in range(L)]
        self.device = device

    def predict(self, x):
        """Predict method."""
        m, d = x.shape
        y_hat = torch.zeros(m, device=self.device)
        for i, learner in enumerate(self.learners):
            xx, xy = x[i, :], torch.zeros((1, 1), device=self.device)
            y_hat[i] = learner(xx, xy)
        return y_hat

    def step(self, x, y, lr=1e-4, weight_decay=1):
        """Step method."""
        m, d = x.shape
        y_hat = self.predict(x)
        loss0 = F.mse_loss(y_hat, y, reduction='sum')
        loss = loss0
        for learner in self.learners:
            for para in learner.parameters():
                loss += para.norm(p=2) * weight_decay
        loss.backward()
        for learner in self.learners:
            for para in learner.parameters():
                para.data.add_(para.grad.data, alpha=-lr)
        return loss0


class RegressorChain:
    """Regressor Chain."""

    def __init__(self, L, baselearner, device=device):
        """__init__ for EvolutionaryRegressorChain."""
        self.order = torch.randperm(L)
        self.learners = [baselearner(device=device) for _ in range(L)]
        self.device = device

    def predict(self, x):
        """Predict method."""
        m, d = x.shape
        y_hat = torch.zeros(m, device=self.device)
        for i, sid in enumerate(self.order):
            if i == 0:
                xx = x[sid, :]
                xy = torch.ones((1, 1), device=self.device) * 0
                y_hat[sid] = self.learners[sid](xx, xy)
            else:
                prev_sid = self.order[i-1]
                xx = x[sid, :]
                xy = torch.ones((1, 1), device=self.device) * \
                    y_hat[prev_sid].item()
                y_hat[sid] = self.learners[sid](xx, xy)
        return y_hat

    def step(self, x, y, lr=1e-4, weight_decay=1):
        """Step method."""
        m, d = x.shape
        y_hat = self.predict(x)
        loss = F.mse_loss(y_hat, y, reduction='sum')
        for learner in self.learners:
            for para in learner.parameters():
                loss += para.norm(p=2) * weight_decay
        loss.backward()
        for learner in self.learners:
            for para in learner.parameters():
                para.data.add_(para.grad.data, alpha=-lr)
        return y_hat


class RegressorChains:
    """Ensemble of Regressor Chains."""

    def __init__(self, n_chains, L, baselearner, device=device):
        """__init__ for RegressorChains."""
        super(RegressorChains, self).__init__()
        self.n_chains = n_chains
        self.chains = [RegressorChain(L, baselearner, device)
                       for _ in range(n_chains)]
        self.device = device

    def predict(self, x):
        """Predict method."""
        m, d = x.shape
        y_hat = torch.zeros((self.n_chains, m), devcie=self.device)
        for i, chain in enumerate(self.chains):
            y_hat[i, :] = chain.predict(x)
        y_hat2 = y_hat.sum(axis=0) / self.n_chains
        return y_hat2

    def step(self, x, y, lr=1e-4, weight_decay=1):
        """Step method."""
        m, d = x.shape
        y_hat = torch.zeros((self.n_chains, m), device=self.device)
        for i, chain in enumerate(self.chains):
            y_hat[i, :] = chain.step(x, y, lr, weight_decay)
        y_hat2 = y_hat.sum(axis=0) / self.n_chains
        loss = F.mse_loss(y_hat2, y, reduction='none')
        return loss


if __name__ == "__main__":
    print('Test SM on TrainStreams')
    print('loading data...')
    train = TrainStreams(device)
    sm_train = SingleModels(8, BaseLearnerTrain)
    for i, data in enumerate(train):
        if i == 10:
            break
        print('  Processing sample {}...'.format(i+1))
        x, y = data
        sm_train.step(x, y)
    print("Test complete.")
    print()
    ###########
    #  split  #
    ###########
    print('Test RC on TrainStreams')
    rc_train = RegressorChains(10, 8, BaseLearnerTrain)
    for i, data in enumerate(train):
        if i == 10:
            break
        print('  Processing sample {}...'.format(i+1))
        x, y = data
        rc_train.step(x, y)
    print("Test complete.")
    print()
    ###########
    #  split  #
    ###########
    print('Test SM on WeatherStreams')
    print('loading data...')
    weather = WeatherStreams(device)
    sm_weather = SingleModels(10, BaseLearnerWeather)
    for i, data in enumerate(weather):
        if i == 10:
            break
        print('  Processing sample {}...'.format(i+1))
        x, y = data
        sm_weather.step(x, y)
    print("Test complete.")
    print()
    ###########
    #  split  #
    ###########
    print('Test RC on WeatherStreams')
    rc_weather = RegressorChains(10, 10, BaseLearnerWeather)
    for i, data in enumerate(weather):
        if i == 10:
            break
        print('  Processing sample {}...'.format(i+1))
        x, y = data
        rc_weather.step(x, y)
    print("Test complete.")
    print()
    ###########
    #  split  #
    ###########
    print('Test SM on SensorStreams')
    print('loading data...')
    sensor = SensorStreams(device)
    sm_sensor = SingleModels(6, BaseLearnerSensor)
    for i, data in enumerate(sensor):
        if i == 10:
            break
        print('  Processing sample {}...'.format(i+1))
        x, y = data
        sm_sensor.step(x, y)
    print("Test complete.")
    print()
    ###########
    #  split  #
    ###########
    print('Test RC on SensorStreams')
    rc_sensor = RegressorChains(10, 6, BaseLearnerSensor)
    for i, data in enumerate(sensor):
        if i == 10:
            break
        print('  Processing sample {}...'.format(i+1))
        x, y = data
        rc_sensor.step(x, y)
    print("Test complete.")
