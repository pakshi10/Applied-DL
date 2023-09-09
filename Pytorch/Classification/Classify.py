import sklearn
from sklearn.datasets import make_circles  
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("Pytorch version: ", torch.__version__)

# Generate data
n_samples = 1000
X, Y = make_circles(n_samples=n_samples, noise=0.1, random_state=1, factor=0.2)

# Print the shape of X and Y
print("Shape of X: ", X.shape)
print("Shape of Y: ", Y.shape)

# Plot the data
def plot_data(X, Y, title=None):
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
    plt.show()
    plt.savefig(f'{title}.png')
    plt.close()

plot_data(X, Y, title='Original data')

# Create a function to convert numpy array to torch tensor
def convert_to_tensor(X, Y):
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    #print first 5 values of X and Y
    print("First 5 values of X: ", X[:5])
    print("First 5 values of Y: ", Y[:5])
    return X, Y

X, Y = convert_to_tensor(X, Y)

#split the data into train and test using sklearn
def split_train_test(X, Y, train_ratio=0.8):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_ratio)
    print("Length of train data: ", len(X_train))
    print("Length of test data: ", len(X_test))
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_train_test(X, Y, train_ratio=0.8)

#plot the train and test data
plot_data(X_train, Y_train, title='Train data')
plot_data(X_test, Y_test, title='Test data')

#Build the model
class Classification(nn.Module):
    def __init__(self):
        super().__init__()
        #parameters that need to be learned
        self.layer1 = nn.Linear(in_features=2,out_features=5) #Hidden layer has 5 neurons
        self.layer2 = nn.Linear(in_features=5,out_features=1) #Takes 5 inputs and gives 1 output 
    
    def forward(self, x):
        return self.layer2(self.layer1(x))
    
#to device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
#model instance to device
model = Classification().to(device)

model_1 = nn.Sequential(
    nn.Linear(2, 5),
    nn.Linear(5, 1),
).to(device)

#print the model parameters
print("Model parameters: ", model.state_dict())
#Make data to device
X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)

#Make predictions
with torch.inference_mode():
    untrained_preds = model(X_train)
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(Y_test)}, Shape: {Y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{Y_test[:10]}")
#model predictions are in probability format, convert them to binary format

# loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss() #Binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) #Stochastic gradient descent

#Calculate accuracy
# Calculate accuracy (a classification metric)
def calc_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


#train the model
epochs = 3000
# Build training and evaluation loop
for epoch in range(epochs):
    ### Training
    model.train()

    # 1. Forward pass (model outputs raw logits)
    y_logits = model(X_train).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls
  
    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
    #                y_train) 
    loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                   Y_train) 
    acc = calc_accuracy(y_true=Y_train, 
                      y_pred=y_pred) 

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            Y_test)
        test_acc = calc_accuracy(y_true=Y_test,
                               y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary




# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, Y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, Y_test)
#save the plot
plt.savefig('Decision_boundary.png')

#Let's imporve the model
# Build a model with a non-linear activation function
'''
5. Improving a model (from a model perspective)

Let's try to fix our model's underfitting problem.

Focusing specifically on the model (not the data), there are a few ways we could do this.
Model improvement technique* 	What does it do?
Add more layers 	Each layer potentially increases the learning capabilities of the model with each layer being able to learn some kind of new pattern in the data, more layers is often referred to as making your neural network deeper.
Add more hidden units 	Similar to the above, more hidden units per layer means a potential increase in learning capabilities of the model, more hidden units is often referred to as making your neural network wider.
Fitting for longer (more epochs) 	Your model might learn more if it had more opportunities to look at the data.
Changing the activation functions 	Some data just can't be fit with only straight lines (like what we've seen), using non-linear activation functions can help with this (hint, hint).
Change the learning rate 	Less model specific, but still related, the learning rate of the optimizer decides how much a model should change its parameters each step, too much and the model overcorrects, too little and it doesn't learn enough.
Change the loss function 	Again, less model specific but still important, different problems require different loss functions. For example, a binary cross entropy loss function won't work with a multi-class classification problem.
'''


