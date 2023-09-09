# Import dependencies
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import plot_predictions, plot_decision_boundary

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
    n_features=NUM_FEATURES, # X features
    centers=NUM_CLASSES, # y labels 
    cluster_std=1.5, # give the clusters a little shake up (try changing this to 1.0, the default)
    random_state=RANDOM_SEED
)

# 2. Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
print(X_blob[:5], y_blob[:5])

# 3. Split into train and test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
    y_blob,
    test_size=0.2,
    random_state=RANDOM_SEED
)

# 4. Plot data
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu);
#save the plot
print("Saving the plot...")
plt.savefig('multiclass_data.png')
plt.close()



# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device


# 5. Create a multi-class classification model
class MultiClassClassification(torch.nn.Module):
    def __init__(self, input_dim,output_dim, hidden_dim=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            #nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            #nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
        )
    def forward(self, x):
        logits = self.linear_layer_stack(x)
        return logits

# 6. Instantiate the model
model_4 = MultiClassClassification(input_dim=NUM_FEATURES, output_dim=NUM_CLASSES)
model_4.to(device)
print(model_4)

# Create loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_4.parameters(), 
                            lr=0.1) # exercise: try changing the learning rate here and seeing what happens to the model's performance

# Perform a single forward pass on the data (we'll need to put it to the target device for it to work)
print(model_4(X_blob_train.to(device))[:5])

# logits -> probabilities -> class predictions
print(torch.softmax(model_4(X_blob_train.to(device))[:5], dim=1))
print(torch.argmax(torch.softmax(model_4(X_blob_train.to(device))[:5], dim=1), dim=1))

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

# Train the model
# Fit the model
torch.manual_seed(42)

# Set number of epochs
epochs = 100

# Put data to target device
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    ### Training
    model_4.train()

    # 1. Forward pass
    y_logits = model_4(X_blob_train) # model outputs raw logits 
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # go from logits -> prediction probabilities -> prediction labels
    # print(y_logits)
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_blob_train) 
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_4.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model_4(X_blob_test)
      test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
      # 2. Calculate test loss and accuracy
      test_loss = loss_fn(test_logits, y_blob_test)
      test_acc = accuracy_fn(y_true=y_blob_test,
                             y_pred=test_pred)

    # Print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%") 

# Test the model on the test data


# Make predictions
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)
y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
# Compare first 10 model preds and test labels
print(f"Predictions: {y_preds[:10]}\nLabels: {y_blob_test[:10]}")
print(f"Test accuracy: {accuracy_fn(y_true=y_blob_test, y_pred=y_preds)}%")



plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
#save the plot
plt.savefig('multiclass_decision_boundary.png')

# Generate metrics for the model using torchmetrics
import torchmetrics
from torchmetrics.functional import accuracy, precision, recall
# Setup metric and make sure it's on the target device
acc = torchmetrics.Accuracy(task='multiclass', num_classes=4).to(device)
prec = torchmetrics.Precision(task='multiclass',num_classes=int(NUM_CLASSES)).to(device)
rec = torchmetrics.Recall(task='multiclass',num_classes=NUM_CLASSES).to(device)
F1 = torchmetrics.F1Score(task='multiclass',num_classes=NUM_CLASSES).to(device)
confusion_matrix = torchmetrics.ConfusionMatrix(task='multiclass',num_classes=NUM_CLASSES).to(device)
#calc acc,prec,rec,F1,confusion metric
#print the metrics
print("Metrics for the model:")
print(f"Accuracy: {acc(y_preds, y_blob_test)}")
print(f"Precision: {prec(y_preds, y_blob_test)}")
print(f"Recall: {rec(y_preds, y_blob_test)}")
print(f"F1: {F1(y_preds, y_blob_test)}")
print(f"Confusion matrix: \n{confusion_matrix(y_preds, y_blob_test)}")

