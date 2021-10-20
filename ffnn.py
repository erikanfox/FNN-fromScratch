"""Binary classifier using PyTorch.
Erika Fox
Patrick Wang, 2021
"""
from abc import ABC
from re import X

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid

from gen_data import gen_simple, gen_xor


class FFNN(ABC):
    """Feed-forward neural network."""

    def __init__(self, learning_rate=1e-2, device='cpu'):
        """Initialize."""
        self.device = device
        self.learning_rate = learning_rate

        if self.device == 'cuda':
            self.net.cuda()

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), 
            lr=self.learning_rate,
        )

    def predict(self, X_t):
        """Predict."""
        return self.net(X_t)

    def update_network(self, y_hat, Y_t):
        """Update weights."""
        self.optimizer.zero_grad()
        loss = self.loss_func(y_hat, Y_t)
        loss.backward()
        self.optimizer.step()
        self.training_loss.append(loss.item())

    def calculate_accuracy(self, y_hat_class, Y):
        """Calculate accuracy."""
        return np.sum(Y.reshape(-1, 1) == y_hat_class) / len(Y)

    def train(self, X, Y, n_iters=1000):
        """Train network."""
        self.training_loss = []
        self.training_accuracy = []

        X_t = torch.FloatTensor(X).to(device=self.device)
        Y = Y.reshape(-1, 1)
        Y_t = torch.FloatTensor(Y).to(device=self.device)

        for _ in range(n_iters):
            y_hat = self.predict(X_t)
            self.update_network(y_hat, Y_t)
            y_hat_class = np.where(y_hat < 0.5, 0, 1)
            accuracy = self.calculate_accuracy(y_hat_class, Y)
            self.training_accuracy.append(accuracy)

    def plot_training_progress(self):
        """Plot training progress."""
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        ax[0].plot(self.training_loss)
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Training Loss')

        ax[1].plot(self.training_accuracy)
        ax[1].set_ylabel('Classification Accuracy')
        ax[1].set_title('Training Accuracy')

        plt.tight_layout()
        plt.show()

    def plot_testing_results(self, X_test, Y_test):
        """Plot testing results."""
        X_t = torch.FloatTensor(X_test).to(device=self.device)
        y_hat_test = self.predict(X_t)
        y_hat_test_class = np.where(y_hat_test < 0.5, 0, 1)

        # Plot the decision boundary
        # Determine grid range in x and y directions
        x_min, x_max = X_test[:, 0].min() - 0.1, X_test[:, 0].max() + 0.1
        y_min, y_max = X_test[:, 1].min() - 0.1, X_test[:, 1].max() + 0.1

        # Set grid spacing parameter
        spacing = min(x_max - x_min, y_max - y_min) / 100

        # Create grid
        XX, YY = np.meshgrid(
            np.arange(x_min, x_max, spacing),
            np.arange(y_min, y_max, spacing)
        )

        # Concatenate data to match input
        data = np.hstack((
            XX.ravel().reshape(-1, 1),
            YY.ravel().reshape(-1, 1),
        ))

        # Pass data to predict method
        data_t = torch.FloatTensor(data).to(device=self.device)
        db_prob = self.predict(data_t)

        clf = np.where(db_prob < 0.5, 0, 1)

        Z = clf.reshape(XX.shape)

        print("Test Accuracy {:.2f}%".format(
            self.calculate_accuracy(y_hat_test_class, Y_test) * 100)
        )

        plt.figure(figsize=(12, 8))
        plt.contourf(XX, YY, Z, cmap=plt.cm.RdYlBu, alpha=0.5)
        plt.scatter(
            X_test[:, 0], X_test[:, 1],
            c=Y_test,
            cmap=plt.cm.RdYlBu,
        )
        plt.show()


class BinaryLinear(FFNN):
    """Linear FFNN for binary classification."""
    def __init__(self, n_input, **kwargs):
        """Initialize."""
        self.n_input_dim = n_input
        self.n_output = 1
        l= nn.Linear(self.n_input_dim, 100)
        l2= nn.Linear(100, self.n_output)
        
        self.net = nn.Sequential(
            l,
            nn.Sigmoid(),
            l2,
            nn.Sigmoid(),
        )

        #print(list(l.parameters()))
        #print(list(l2.parameters()))

        self.loss_func = nn.BCELoss()

        super().__init__(**kwargs)

def ffnn_man(params,X_test, Y_test):
    layer1=np.array(params[0].data)
    bias1=np.array(params[1].data)
    layer2=np.array(params[2].data)
    bias2= np.array(params[3].data)


    yhat=np.zeros(len(X_test))
    for i in range(len(X_test)):
        z1=np.array(X_test[i,:]).T
        z2= np.add((layer1 @ z1),bias1)
        m= 1/(1 + np.exp(-z2))
        m2= np.add((layer2 @ m), bias2)
        ans = 1/(1 + np.exp(-m2))
        yhat[i]=ans
    yval=np.where(yhat > 0.5,1,0)   
    ret=(sum(Y_test==yval)/len(Y_test))*100
    return f"The accuracy of my neural network is {ret}%"

def main():
    """Run experiment."""
    n_dims = 2
    X, Y = gen_xor(400)

    # Split into test and training data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.25,
    )
    net = BinaryLinear(n_dims)
    net.train(X_train, Y_train)
    net.plot_training_progress()
    net.plot_testing_results(X_test, Y_test)
    params=list(net.net.parameters())
    x = ffnn_man(params, X_test, Y_test)
    print(x)



if __name__ == "__main__":
    main()
