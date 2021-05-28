# -*- coding: utf-8 -*-
"""Data loader for Evoluationary Regressor Chains.

Data loaders, all datasets are indexed in size of (t, m, d),
where t is the timestamp of data,
m is the number of data streams,
and d is the dimension of the feature
"""
import pandas as pd
import torch
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SyntheticStreams(Dataset):
    """SyntheticStreams dataset."""

    def __init__(self, streams_id, device=device):
        """__init__ for SyntheticStreams."""
        super(SyntheticStreams, self).__init__()
        df = pd.read_csv('data/multistream.synthetic{}.csv'.format(streams_id))
        _, d = df.shape
        df_npy = df.to_numpy(dtype='float32').reshape((50000, -1, d))
        self.x = torch.from_numpy(df_npy[:, :, :-1]).to(device)
        self.y = torch.from_numpy(df_npy[:, :, -1]).to(device)
        del df, df_npy

    def __len__(self):
        """Methods for emulating a container type."""
        return 50000

    def __getitem__(self, key):
        """Methods for emulating a container type."""
        return self.x[key, :, :], self.y[key, :]


class TrainStreams(Dataset):
    """Train dataset."""

    def __init__(self, device=device):
        """__init__ for TrainStreams."""
        super(TrainStreams, self).__init__()
        df = pd.read_csv('data/multistream.train.csv')
        self.x = torch.zeros((62682, 8, 12), device=device)
        self.y = torch.zeros((62682, 8), device=device)
        x_period = df['PERIODS'].to_numpy()
        x_stations = df[['STATION_NAME',
                         'TRIP_ORGIN_STATION',
                         'TRIP_DESTINATION_STATION']].to_numpy()
        x_platforms = df['PLANNED_ARRIVAL_PLATFORM'].to_numpy()
        for i in range(8):
            self.x[:, i, 0] = torch.from_numpy(x_period).to(device)
            self.x[:, i, 1:4] = torch.from_numpy(x_stations).to(device)
            self.x[:, i, 4] = torch.from_numpy(x_platforms).to(device)
            x_others = df[['SEGMENT_DIRECTION_NAME', 'TRIP_DURATION_SEC',
                           'DWELL_TIME', 'DAY_OF_YEAR', 'DAY_OF_WEEK', 'TIME',
                           'CAR{}_PSNGLD_ARRIVE'.format(i+1)]].to_numpy()
            y = df['CAR{}_PSNGLD_DEPART'.format(i+1)].to_numpy()
            self.x[:, i, 5:] = torch.from_numpy(x_others).to(device)
            self.y[:, i] = torch.from_numpy(y).to(device)
        del df

    def __len__(self):
        """Methods for emulating a container type."""
        return 62682

    def __getitem__(self, key):
        """Methods for emulating a container type."""
        return self.x[key, :, :], self.y[key, :]


class WeatherStreams(Dataset):
    """Weather dataset."""

    def __init__(self, device=device):
        """__init__ for Weather."""
        super(WeatherStreams, self).__init__()
        df = pd.read_csv('data/multistream.weather.csv')
        ser_label = df.pop('w10m_obs')
        self.x = torch.zeros((49728, 10, 8), device=device)
        self.y = torch.zeros((49728, 10), device=device)
        for i in range(10):
            i1 = i * 49728
            i2 = i1 + 49728
            self.x[:, i, :] = \
                torch.from_numpy(df.iloc[i1:i2, :].to_numpy()).to(device)
            self.y[:, i] = \
                torch.from_numpy(ser_label.iloc[i1:i2].to_numpy()).to(device)
        del df

    def __len__(self):
        """Methods for emulating a container type."""
        return 49728

    def __getitem__(self, key):
        """Methods for emulating a container type."""
        return self.x[key, :, :], self.y[key, :]


class SensorStreams(Dataset):
    """Sensor dataset."""

    def __init__(self, device=device):
        """__init__ for SensorStreams."""
        super(SensorStreams, self).__init__()
        df = pd.read_csv('data/multistream.sensor.csv')
        self.x = torch.zeros((20844, 6, 3), device=device)
        self.y = torch.zeros((20844, 6), device=device)
        for i in range(6):
            i1 = 4 * i
            i2 = i1 + 3
            self.x[:, i, :] = \
                torch.from_numpy(df.iloc[:, i1:i2].to_numpy()).to(device)
            self.y[:, i] = \
                torch.from_numpy(df.iloc[:, i2].to_numpy()).to(device)
        del df

    def __len__(self):
        """Methods for emulating a container type."""
        return 20844

    def __getitem__(self, key):
        """Methods for emulating a container type."""
        return self.x[key, :, :], self.y[key, :]


if __name__ == "__main__":
    print('Test SydneyTrain dataset...')
    train = TrainStreams()
    for i, data in enumerate(train):
        x, y = data
        print(x.shape, y.shape)
        if i == 10:
            break
    print('Test complete!')
    print()
    print('Test Weather dataset...')
    weather = WeatherStreams()
    for i, data in enumerate(weather):
        x, y = data
        print(x.shape, y.shape)
        if i == 10:
            break
    print('Test complete!')
    print()
    print('Test Sensor dataset...')
    sensor = SensorStreams()
    for i, data in enumerate(sensor):
        x, y = data
        print(x.shape, y.shape)
        if i == 10:
            break
    print('Test complete!')
