import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#split the data into train and test
def split_train_test(X, Y, train_ratio=0.8):
    train_size = int(X.shape[0] * train_ratio)
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]
    print("Length of train data: ", len(X_train))
    print("Length of test data: ", len(X_test))
    return X_train, Y_train, X_test, Y_test

def plot_data(X, Y, Y_pred=None, title=None):
    plt.plot(X.numpy(), Y.numpy(), 'r', label='Original data')
    if Y_pred is not None:
        plt.plot(X.numpy(), Y_pred.detach().numpy(), 'b-', label='Fitted line(Model)')
    #plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()
    plt.savefig(f'{title}.png')
    plt.close()

def plot_result(X, Y, Y_pred=None, title=None):
    plt.plot(X.numpy(), Y.numpy(), 'r', label='Original data')
    if Y_pred is not None:
        plt.plot(X.numpy(), Y_pred.detach().numpy(), 'b-', label='Fitted line(Model)')
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()
    plt.savefig('result.png')
    plt.close()

#plot the train and test loss
def plot_loss(trainloss,testloss):
    plt.plot(trainloss, label='Train loss')
    plt.plot(testloss, label='Test loss')
    plt.legend()
    plt.show()
    plt.savefig('loss.png')
    plt.close()


#Build the model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        #parameters that need to be learned
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True,dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True,dtype=torch.float32))
    def forward(self, x):
        return x * self.weights + self.bias
    


# Hyper-parameters
weight_true = 3.5
bias_true = 2.5
learning_rate = 0.001

# Create random input data
X = torch.arange(start=0, end=5, step=0.02).unsqueeze(1)
Y = weight_true * X + bias_true

print(X[:10])
print(Y[:10])

#split the data into train and test
X_train, Y_train, X_test, Y_test = split_train_test(X, Y, train_ratio=0.8)

# Create a random seed
torch.manual_seed(100)
#create a instance of the model
model = LinearRegression()
print(model)
#print parameters
for name, param in model.named_parameters():
    print(name, param)

#Testing the model predictions using inference faster
with torch.inference_mode():
    Y_pred = model(X_test)
# with torch.no_grad():
#     Y_pred = model(X_test)
#plot the trainning data and see the test data
plot_data(X_train, Y_train, title='Train Data')
plot_data(X_test, Y_test, Y_pred, title='Test Data')

#loss function
criterion = nn.MSELoss()
#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#train the model
num_epochs = 200
train_loss = []
for epoch in range(num_epochs):
    #forward pass
    Y_pred = model(X_train)
    #compute loss
    loss = criterion(Y_pred, Y_train)
    #backward pass
    loss.backward()
    #update parameters
    optimizer.step()
    #zero grad before new step
    optimizer.zero_grad()
    #print loss
    print('epoch {}, loss {}'.format(epoch, loss.item()))
    #store the loss and plot it
    train_loss.append(loss.item())
    #plot the result
    plot_result(X_train, Y_train, Y_pred.detach(), title=f'Epoch {epoch}')

#test the model
test_loss = []
with torch.inference_mode():
    Y_pred = model(X_test)
    loss = criterion(Y_pred, Y_test)
    #store the loss and plot it
    test_loss.append(loss.item())
    print('Test loss: ', loss.item())
    plot_data(X_test, Y_test, Y_pred.detach(), title='Test Data')

#plot the loss
plot_loss(train_loss,test_loss)

#save the model
torch.save(model.state_dict(), 'model.pth')
print('Saved PyTorch Model State to model.pth')
