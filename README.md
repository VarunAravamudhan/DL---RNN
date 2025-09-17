

# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM

To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.



## THEORY

Stock market forecasting is a **time series prediction problem** where future stock prices are estimated based on past historical data.

* **Time Series Data**: Sequential data points recorded at fixed time intervals (e.g., daily stock closing prices).
* **Recurrent Neural Networks (RNNs)**:

  * Unlike feed-forward neural networks, RNNs can maintain a "memory" of past inputs.
  * Each output depends not only on the current input but also on previous hidden states, making them suitable for sequential data like stock prices.
* **Challenges in Stock Prediction**:

  * Stock markets are volatile and influenced by multiple factors (economic, social, political).
  * RNNs help identify hidden patterns and trends in sequential financial data.
* **Improved Variants**:

  * **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** mitigate vanishing gradient problems and capture long-term dependencies.

Thus, RNN-based models are widely applied for forecasting financial time series such as stock prices.



## Neural Network Model

**Model Flow:**

```
Input Sequence (Closing Prices for N past days)
        ↓
     RNN / LSTM Layer(s)
        ↓
 Fully Connected Dense Layer
        ↓
   Predicted Stock Price
```



## DESIGN STEPS

### STEP 1: Import Libraries

Import necessary Python libraries such as **NumPy, Pandas, Matplotlib, PyTorch, and Scikit-learn** for data preprocessing, model building, and evaluation.

### STEP 2: Load and Preprocess Data

* Load historical stock data (CSV or API source).
* Extract the **‘Close’** price column.
* Normalize values using **MinMaxScaler** to scale between \[0,1] for better convergence.
* Convert time series into supervised learning format by creating sequences (past N days → next day).

### STEP 3: Train-Test Split

* Divide dataset into **80% training** and **20% testing**.
* Convert data into PyTorch tensors for RNN input.

### STEP 4: Build RNN Model

* Define RNN layers with input size, hidden size, and number of layers.
* Add a **fully connected layer** to map hidden state to output.
* Use activation functions to ensure non-linearity.

### STEP 5: Train the Model

* Define **loss function**: Mean Squared Error (MSE).
* Define **optimizer**: Adam for adaptive learning.
* Train for multiple epochs while updating weights.
* Track training loss for convergence.

### STEP 6: Evaluate and Visualize

* Use trained RNN to predict stock prices on test data.
* Inverse-transform scaled predictions back to original prices.
* Plot:

  1. **Training Loss vs Epochs**.
  2. **Actual vs Predicted Stock Prices**.
* Compare predictions with true values.



## PROGRAM

### Name: Varun A 

### Register Number: 212224240178

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

## Step 1: Load and Preprocess Data
# Load training and test datasets
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')

# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)


x_train.shape, y_train.shape, x_test.shape, y_test.shape

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

## Step 2: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1,hidden_size=64,num_layers=2,output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size,output_size)
    def forward(self, x):
        out,_=self.rnn(x)
        out=self.fc(out[:,-1,:])
        return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

!pip install torchinfo

from torchinfo import summary

# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

## Step 3: Train the Model

num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Plot training loss
print('Name: Varun A                ')
print('Register Number:  212224240178   ')
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()


## Step 4: Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name:  Varun A               ')
print('Register Number: 212224240178    ')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')
```



## OUTPUT

### Training Loss Over Epochs Plot

<img width="937" height="447" alt="image" src="https://github.com/user-attachments/assets/18026d8f-f51f-453a-8666-5acd09167815" />

### True Stock Price, Predicted Stock Price vs Time

<img width="893" height="465" alt="image" src="https://github.com/user-attachments/assets/7fd1fa2f-2bbd-45f2-a3da-ff4bb9a1c6f5" />

### Predictions

<img width="255" height="72" alt="image" src="https://github.com/user-attachments/assets/351acf44-bbb9-4c00-bca9-8f53ec73474c" />



## RESULT

Thus, a **Recurrent Neural Network model** was successfully developed to predict stock prices using historical closing price data. The model was trained, evaluated, and visualized with actual vs predicted results, showing that RNNs can capture sequential dependencies in financial time series.


