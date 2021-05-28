# -*- coding: utf-8 -*-
"""Base learners.

Base learners of data streams
"""
import torch
import torch.nn as nn

from ..utils.loader import TrainStreams
from ..utils.loader import WeatherStreams
from ..utils.loader import SensorStreams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseLearnerSynthetic(nn.Module):
    """Base learner for SyntheticStreams.

    A simple linear model.
    """

    def __init__(self, device=device):
        """__init__ for BaseLearnerSynthetic."""
        super(BaseLearnerSynthetic, self).__init__()
        self.fcx = nn.Linear(4, 1, bias=False)
        self.fcy = nn.Linear(1, 1)
        nn.init.kaiming_uniform_(self.fcx.weight)
        nn.init.kaiming_uniform_(self.fcy.weight)
        nn.init.zeros_(self.fcy.bias)
        self.to(device)

    def forward(self, x, y):
        """Forward method."""
        if len(x.shape) == 1:
            x = x.view(1, -1)
        x1 = self.fcx(x)
        x2 = self.fcy(y)
        xx = x1 + x2
        return xx


class BaseLearnerTrain(nn.Module):
    """Base learner for TrainStreams.

    A simple linear model with embedding layers.
    """

    def __init__(self, device=device):
        """__init__ for BaseLearnerTrain."""
        super(BaseLearnerTrain, self).__init__()
        self.embedding_period = nn.Embedding(4, 2)
        self.embedding_stations = nn.Embedding(162, 3)
        self.embedding_platforms = nn.Embedding(460, 3)
        self.fc_period = nn.Linear(2, 1, bias=False)
        self.fc_stations = nn.Linear(9, 1, bias=False)
        self.fc_platforms = nn.Linear(3, 1, bias=False)
        self.fcn = nn.Linear(7, 1, bias=False)
        self.fcy = nn.Linear(1, 1)
        nn.init.kaiming_uniform_(self.fc_period.weight)
        nn.init.kaiming_uniform_(self.fc_stations.weight)
        nn.init.kaiming_uniform_(self.fc_platforms.weight)
        nn.init.kaiming_uniform_(self.fcn.weight)
        nn.init.kaiming_uniform_(self.fcy.weight)
        nn.init.zeros_(self.fcy.bias)
        self.to(device)

    def forward(self, x, y):
        """Forward method."""
        if len(x.shape) == 1:
            x = x.view(1, -1)
        n, d = x.shape
        x1 = self.embedding_period(x[:, 0].to(torch.int32))
        x1 = self.fc_period(x1.view(n, -1))
        x2 = self.embedding_stations(x[:, 1:4].to(torch.int32))
        x2 = self.fc_stations(x2.view(n, -1))
        x3 = self.embedding_platforms(x[:, 4].to(torch.int32))
        x3 = self.fc_platforms(x3.view(n, -1))
        x4 = self.fcn(x[:, 5:])
        x5 = self.fcy(y)
        xx = x1 + x2 + x3 + x4 + x5
        return xx


class BaseLearnerWeather(nn.Module):
    """Base learner for WeatherStreams.

    A simple linear model.
    """

    def __init__(self, device=device):
        """__init__ for BaseLearnerWeather."""
        super(BaseLearnerWeather, self).__init__()
        self.fcx = nn.Linear(8, 1, bias=False)
        self.fcy = nn.Linear(1, 1)
        nn.init.kaiming_uniform_(self.fcx.weight)
        nn.init.kaiming_uniform_(self.fcy.weight)
        nn.init.zeros_(self.fcy.bias)
        self.to(device)

    def forward(self, x, y):
        """Forward method."""
        if len(x.shape) == 1:
            x = x.view(1, -1)
        x1 = self.fcx(x)
        x2 = self.fcy(y)
        xx = x1 + x2
        return xx


class BaseLearnerSensor(nn.Module):
    """Base learner for SensorStreams.

    A simple linear model.
    """

    def __init__(self, device=device):
        """__init__ for BaseLearnerSensor."""
        super(BaseLearnerSensor, self).__init__()
        self.fcx = nn.Linear(3, 1, bias=False)
        self.fcy = nn.Linear(1, 1)
        nn.init.kaiming_uniform_(self.fcx.weight)
        nn.init.kaiming_uniform_(self.fcy.weight)
        nn.init.zeros_(self.fcy.bias)
        self.to(device)

    def forward(self, x, y):
        """Forward method."""
        if len(x.shape) == 1:
            x = x.view(1, -1)
        x1 = self.fcx(x)
        x2 = self.fcy(y)
        xx = x1 + x2
        return xx


if __name__ == "__main__":
    sydtrain = TrainStreams()
    x, y = sydtrain[:10]
    print("Test BaseLearnerTrain forward method...")
    learner = BaseLearnerTrain()
    y_predict = learner(x[:, 0, :], y[:, 0:1])
    print("y_predict = ", y_predict)
    print("Test complete.")
    print()
    weather = WeatherStreams()
    x, y = weather[:10]
    print("Test BaseLearnerWeather forward method...")
    learner = BaseLearnerWeather()
    y_predict = learner(x[:, 0, :], y[:, 0:1])
    print("y_predict = ", y_predict)
    print("Test complete.")
    print()
    sensor = SensorStreams()
    x, y = sensor[:10]
    print("Test BaseLearnerSensor forward method...")
    learner = BaseLearnerSensor()
    y_predict = learner(x[:, 0, :], y[:, 0:1])
    print("y_predict = ", y_predict)
    print("Test complete.")
