import sklearn
from sklearn.datasets import make_circles  
from sklearn.model_selection import train_test_split
from helper_functions import plot_predictions, plot_decision_boundary
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

#to device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
#Make data to device
X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        #parameters that need to be learned
        self.layer1 = nn.Linear(in_features=2,out_features=16) 
        self.layer2 = nn.Linear(in_features=16,out_features=32) 
        self.layer3 = nn.Linear(in_features=32,out_features=1) 
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))
    
model = CircleModel()
model.to(device)
print(model)

#loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#train and test the model
def train_and_test(model, criterion, optimizer, X_train, Y_train, X_test, Y_test, epochs=100):
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        #train the model
        model.train()
        optimizer.zero_grad()
        Y_pred = model(X_train)
        loss = criterion(Y_pred.squeeze(), Y_train)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        #test the model
        model.eval()
        with torch.no_grad():
            Y_pred = model(X_test)
            loss = criterion(Y_pred.squeeze(), Y_test)
            test_loss.append(loss.item())
        if (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch+1}/{epochs}, Train loss: {train_loss[-1]:.4f}, Test loss: {test_loss[-1]:.4f}')
    return train_loss, test_loss

train_loss, test_loss = train_and_test(model, criterion, optimizer, X_train, Y_train, X_test, Y_test, epochs=100)




# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, Y_train) # model_1 = no non-linearity
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, Y_test) # model_3 = has non-linearity
#save the plot
plt.savefig('Non_linear_classify.png')

# Plot the loss
plt.figure(figsize=(12, 6))
plt.plot(train_loss, label="Train loss")
plt.plot(test_loss, label="Test loss")
plt.legend()
plt.show()
#save the plot
plt.savefig('Non_linear_classify_loss.png')

#save the model
torch.save(model.state_dict(), 'Non_linear_classify_model.pth')

