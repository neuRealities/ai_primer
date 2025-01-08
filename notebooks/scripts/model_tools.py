"""Module to handle model functions"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define our model, with parameterized loss function and optimizer
def train_validate(model, loss_fn, optimizer, trainloader, testloader, device, n_epochs:int=25, flatten:bool = False):
    """Train for number of epochs"""
    train_losses = []
    test_losses = []

    for epoch in range(n_epochs):
        model.train() # Set mode to training - Dropouts will be used here
        train_epoch_loss = 0

        # Iterate over each batch of the training data
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            # flatten the images to batch_size x input features
            if flatten:
                images = images.view(images.shape[0], -1)

            # Clear the gradients from the previous iteration
            optimizer.zero_grad()

            # Forward pass: Pass the images through the model to get the predicted outputs
            outputs = model(images)

            # Compute the loss between the predicted outputs and the true labels
            train_batch_loss = loss_fn(outputs, labels)

            # Backward pass: Compute the gradient of the loss w.r.t. model parameters
            train_batch_loss.backward()

            # Update the model parameters
            optimizer.step()
            train_epoch_loss += train_batch_loss.item()

        # One epoch of training complete
        # calculate average training epoch loss
        train_epoch_loss = train_epoch_loss/len(trainloader)

        # Now Validate on testset
        with torch.no_grad():
            test_epoch_acc = 0
            test_epoch_loss = 0
            model.eval() # Set mode to eval - Dropouts will NOT be used here
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)                    
                # Flatten images to batch_size x input features
                if flatten:
                    images = images.view(images.shape[0], -1)
                # Make predictions 
                test_outputs = model(images)
                # Calculate test loss
                test_batch_loss = loss_fn(test_outputs, labels)
                test_epoch_loss += test_batch_loss

                # Get probabilities, extract the class associated with highest probability
                proba = torch.exp(test_outputs)
                _, pred_labels = proba.topk(1, dim=1)

                # Compare actual labels and predicted labels
                result = pred_labels == labels.view(pred_labels.shape)
                batch_acc = torch.mean(result.type(torch.FloatTensor))
                test_epoch_acc += batch_acc.item()

            # One epoch of training and validation done
            # Calculate average testing epoch loss
            test_epoch_loss = test_epoch_loss/len(testloader)
            test_epoch_loss = test_epoch_loss.cpu() # To be able to plot it
            # Calculate accuracy as correct_pred/total_samples
            test_epoch_acc = test_epoch_acc/len(testloader)
            # Save epoch losses for plotting
            train_losses.append(train_epoch_loss)
            test_losses.append(test_epoch_loss)
            # Print stats for this epoch
            print(f'Epoch: {epoch:02} -> train_loss: {train_epoch_loss:.10f}, val_loss: {test_epoch_loss:.10f}, ',
                    f'val_acc: {test_epoch_acc*100:.2f}%')
    # Finally plot losses
    plt.plot(train_losses, label='train-loss')
    plt.plot(test_losses, label='val-loss')
    plt.legend()
    plt.show()

def test_validate(model, test_dl, device):
    """Validate against test set"""
    with torch.no_grad(): # Ne need to calculate backward pass for test set.
        batch_acc = []
        model.eval()
        for images, labels in test_dl:
            images, labels = images.to(device), labels.to(device)
            # flatten images to batch_size x 784
            images = images.view(images.shape[0], -1)
            # make predictions and get probabilities
            proba = torch.exp(model(images))
            # extract the class associted with highest probability
            _, pred_labels = proba.topk(1, dim=1)
            # compare actual labels and predicted labels
            result = pred_labels == labels.view(pred_labels.shape)
            acc = torch.mean(result.type(torch.FloatTensor))
            batch_acc.append(acc.item())
        print(f'Test Accuracy: {torch.mean(torch.tensor(batch_acc))*100:.2f}%')
        return batch_acc

def test_validate_confusion(model, test_dl, device, classes):
    """Validate against a test set, and display confusion matrix"""
    # Set the model to evaluation mode. This is important as certain layers like 
    # dropout behave differently during training and evaluation.
    model.eval()

    # Lists to store all predictions and true labels
    all_preds = []
    all_labels = []

    # We don't want to compute gradients during evaluation, hence wrap the code inside torch.no_grad()
    with torch.no_grad():
        batch_acc = []
        # Iterate over all batches in the test loader
        for images, labels in test_dl:
            # Transfer images and labels to the computational device (either CPU or GPU)
            images, labels = images.to(device), labels.to(device)

            # Pass the images through the model to get predictions
            outputs = model(images)

            # Get the class with the maximum probability as the predicted class
            _, predicted = torch.max(outputs, 1)

            # Extend the all_preds list with predictions from this batch
            all_preds.extend(predicted.cpu().numpy())

            # Extend the all_labels list with true labels from this batch
            all_labels.extend(labels.cpu().numpy())

            # Compare actual labels and predicted labels
            result = predicted == labels.view(predicted.shape)
            acc = torch.mean(result.type(torch.FloatTensor))
            batch_acc.append(acc.item())

    # Print a classification report which provides an overview of the model's performance for each class
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Compute the confusion matrix using true labels and predictions
    cm = confusion_matrix(all_labels, all_preds)

    # Visualize the confusion matrix using seaborn's heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')  # x-axis label
    plt.ylabel('True Label')       # y-axis label
    plt.title('Confusion Matrix')  # Title of the plot
    plt.show()                     # Display the plot

    print(f'Test Accuracy: {torch.mean(torch.tensor(batch_acc))*100:.2f}%')

def set_fashion_dataset(transform, validation_ratio, batch_size):
    """Return train and test datasets and dataloaders"""
    train_ds  = datasets.FashionMNIST(root="data", train=True,  download=True, transform=transform)
    test_ds   = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
    # Split training set into training and validation
    train_num = len(train_ds)
    indices = list(range(train_num))
    np.random.shuffle(indices)
    split = int(np.floor(validation_ratio * train_num))
    val_idx, train_idx = indices[:split], indices[split:]

    train_dl = DataLoader(train_ds, batch_size=batch_size)
    val_dl   = DataLoader(train_ds, batch_size=batch_size)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    print("Train/Validation/Test Split:", len(train_idx), "/", len(val_idx), "/", len(test_ds))
    classes  = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    return train_ds, test_ds, train_dl, val_dl, test_dl, classes

