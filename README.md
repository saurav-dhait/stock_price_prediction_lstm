# Time Series Forecasting with LSTM

This project demonstrates how to use Long Short-Term Memory (LSTM) networks for time series forecasting using Amazon stock price data. The code includes data preprocessing, model training, validation, and visualization of predictions.

## Project Structure

- `AMZN.csv`: Dataset containing Amazon stock prices.
- `main.py`: Main script containing data preprocessing, model definition, training, and evaluation.

## Requirements

Ensure you have the following Python packages installed:

- numpy
- pandas
- matplotlib
- scikit-learn
- torch

You can install the required packages using the following command:

```sh
pip install -r requirements.txt
```

## Data

The data used in this project is historical stock price data for Amazon (AMZN). The data should be in CSV format with at least two columns: Date and Close. An example of how the data should look:


## Project Structure

- `main.py`: This is the main script that contains the entire pipeline for data preprocessing, model training, and evaluation.
- `AMZN`.csv: The CSV file containing the historical stock price data for Amazon.

## Usage

### Prepare Data

The `prepare_dataframe_for_lstm` function processes the data to create a time series dataset suitable for LSTM. It generates past closing prices as features for predicting the future closing price.

### Define Dataset Class

The `TimeSeriesDataset` class inherits from `torch.utils.data.Dataset` and is used to create dataset objects for training and testing.

### Define LSTM Model

The `LSTM` class defines the architecture of the LSTM neural network.

### Train and Validate Model

The `train_one_epoch` and `validate_one_epoch` functions handle the training and validation process for each epoch.

## Workflow

### Run the Pipeline

The main function orchestrates the entire workflow:

1. **Read and Preprocess Data**: Reads and preprocesses the data.
2. **Split Data**: Splits the data into training and testing sets.
3. **Create DataLoader**: Creates DataLoader objects for batch processing.
4. **Initialize and Train LSTM Model**: Initializes and trains the LSTM model.
5. **Evaluate Model**: Evaluates the model and plots the results.

## Running the code
- To run the project, execute the `main.py` script:

```sh
python main.py
```

## Results
The project will output two plots:
1. Actual vs. Predicted stock prices for the training dataset.

![image](https://github.com/saurav-dhait/stock_price_prediction_lstm/blob/main/img/1.png)

2. Actual vs. Predicted stock prices for the testing dataset.

![image](https://github.com/saurav-dhait/stock_price_prediction_lstm/blob/main/img/2.png)
## Acknowledgements
This project is inspired by various tutorials and resources available for time series forecasting using LSTM networks.