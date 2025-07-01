
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler 




# %%
dataset = pd.read_csv('Telco-Customer-Churn.csv')


# %%
dataset.head(5)

# %%
print(dataset.info())

# %%
print(dataset.describe())

# %%
# clean data and prepare it for ANN model
dataset.drop('customerID', axis = 1, inplace = True)
# here wedrop customer id because it has no use for us then axis = 1 means we drop a column and axis = 0 would mean row and last true = change directly in dataset

# %%
# handle null values
dataset.isnull().sum()
# it means there are no null values


# %%
# using label encoder on each object column
labelencoder = LabelEncoder()
for col in dataset.columns: #loop through eveyr column in datset
    if dataset[col].dtype == 'object':
      dataset[col] = labelencoder.fit_transform(dataset[col])

    

# %%
# scaling all x features 
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %%
# BUILDING ANN

# %%
# input layer
# define architecture how many layers etc
class ChurnModelling(nn.Module):
    def __init__(self, input_size):
        super(ChurnModelling, self).__init__()   #Always needed when subclassing nn.Module.
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    # inside forward pass it goes Input - layer 1 - relu
    # layer1 - layer 2 - relu
    # layer2 - output - sigmoid
    # return prediction
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x
    
    
    
    
        
        

# %%
from torch.utils.data import TensorDataset, DataLoader

# %%
# convert numpy arrays to tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# %%
dataset = TensorDataset(X_tensor, y_tensor)

# %%
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_size = X_train.shape[1]


# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
input_size = X_train.shape[1]
model = ChurnModelling(input_size)


# %%
criterion = nn.BCELoss()


# %%
num_epochs = 50

for epoch in range(num_epochs):
  for batch_X, batch_y in dataloader:
    # forward pass output (what it needs - x)
    
    output = model(batch_X)
    # loss calculator 
    loss = criterion(output, batch_y)

    # optimizer Adam
    optimizer.zero_grad()

    # backpropogation
    loss.backward()
     # optimize
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    

# %%


from sklearn.metrics import accuracy_score

# 1. Switch to evaluation mode
model.eval()

# 2. Disable gradient tracking
with torch.no_grad():
    # 3. Get probabilities from model
    probs = model(X_tensor)

    # 4. Convert probabilities to 0 or 1
    preds = (probs >= 0.5).float()

# 5. Convert to numpy arrays for metric
y_true = y_tensor.numpy()
y_pred = preds.numpy()

# 6. Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc*100:.4f}")

# %%


# %%



