import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    return df


def train_one_epoch(model, epoch, train_loader, device, optimizer, loss_function):
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index + 1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()


def validate_one_epoch(model, test_loader, device, loss_function):
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)
        self.device = device

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def main():
    # reading the data from Amazon stocks csv file
    data = pd.read_csv("AMZN.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # shortening the data so that it only contains Date and Closing column
    data["Date"] = pd.to_datetime(data["Date"])
    data = data[['Date', 'Close']]

    # creating 7 closing columns for LSTM input feed
    lookback = 7
    shifted_df = prepare_dataframe_for_lstm(data, lookback)
    shifted_df_as_np_array = shifted_df.to_numpy()

    # scaling the values so that they range from -1 to 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np_array = scaler.fit_transform(shifted_df_as_np_array)
    x = shifted_df_as_np_array[:, 1:]
    y = shifted_df_as_np_array[:, 0]

    # flipping the 7 closing columns
    x = dc(np.flip(x, axis=1))

    # splitting the data into train and test data
    split_index = int(len(x) * 0.95)
    x_train = x[:split_index]
    x_test = x[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    # adding an extra dimension for LSTM compatibility
    x_train = x_train.reshape((-1, lookback, 1))
    x_test = x_test.reshape((-1, lookback, 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # type casting the data into tensors
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()

    # creating Dataset objects
    train_dataset = TimeSeriesDataset(x_train, y_train)
    test_dataset = TimeSeriesDataset(x_test, y_test)

    # creating data loader objects along with batch size
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # verifying
    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        print(x_batch.shape, y_batch.shape)
        break

    # initialize model, learning rate, loss function, optimizer
    model = LSTM(1, 4, 1, device)
    model.to(device)
    learning_rate = 0.001
    num_epochs = 10
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training the model
    for epoch in range(num_epochs):
        train_one_epoch(model, epoch, train_loader, device, optimizer, loss_function)
        validate_one_epoch(model, test_loader, device, loss_function)



    # changing the scaled values back to their original form and plotting the train dataset
    with torch.no_grad():
        predicted = model(x_train.to(device)).to('cpu').numpy()
    train_predictions = predicted.flatten()
    dummies = np.zeros((x_train.shape[0], lookback + 1))
    dummies[:, 0] = train_predictions
    dummies = scaler.inverse_transform(dummies)
    train_predictions = dc(dummies[:, 0])
    dummies = np.zeros((x_train.shape[0], lookback + 1))
    dummies[:, 0] = y_train.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_train = dc(dummies[:, 0])

    plt.plot(new_y_train, label='Actual Close')
    plt.plot(train_predictions, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()

    # changing the scaled values back to their original form and plotting the test dataset
    test_predictions = model(x_test.to(device)).detach().cpu().numpy().flatten()
    dummies = np.zeros((x_test.shape[0], lookback + 1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)
    test_predictions = dc(dummies[:, 0])
    dummies = np.zeros((x_test.shape[0], lookback + 1))
    dummies[:, 0] = y_test.flatten()
    dummies = scaler.inverse_transform(dummies)

    new_y_test = dc(dummies[:, 0])
    plt.plot(new_y_test, label='Actual Close')
    plt.plot(test_predictions, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
