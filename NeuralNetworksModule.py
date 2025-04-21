import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from matplotlib import pyplot as plt

mp = {0: "A", 1: "B", 2: "C"}
mpp = {'A': 0, 'B': 1, 'C': 2}


# use Model class that uses the neural network Modules
class Model(nn.Module):
    # decide the no of input neurons,
    # no of hidden layers and its no of neurons,
    # no of o/p neurons
    def __init__(self, in_features=2, h1=6, h2=6, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, input):
        at_layer1 = F.relu(self.fc1(input))
        at_layer2 = F.relu(self.fc2(at_layer1))
        final_op = self.out(at_layer2)  # No ReLU here

        return final_op


class MyNN:
    model = Model()

    def __init__(self, dataset):
        _X = []

        for coordinate in dataset["X"]:
            if isinstance(coordinate, (tuple, list)) and len(coordinate) == 2:
                # Normalize the coordinates to the range [0, 1]
                # Assuming the coordinates are in the range [0, 500]
                _X.append([
                    float(coordinate[0]) / 500,
                    float(coordinate[1]) / 500
                ])
            else:
                print("Invalid coordinate format:", coordinate)
        X = pd.DataFrame(_X)
        y = pd.DataFrame(dataset["y"])
        # Debugging
        # print("X shape:", X.shape)
        # print("First few rows of X:", X.head())
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values.flatten(), dtype=torch.long)

        self.losses = []

    def Training(self):
        # training the model:
        torch.manual_seed(45)
        print("Training in progress..")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        epochs = 150

        for i in range(epochs):
            y_pred = self.model.forward(self.X)
            loss = criterion(y_pred, self.y)
            self.losses.append(loss.detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Training Completed..")

    def predict(self, x, y):
        with torch.no_grad():
            new_point = torch.tensor([[x / 500, y / 500]], dtype=torch.float32)
            tnsr = self.model.forward(new_point)
            return mpp[tnsr.argmax().item()]

    def showLoss(self):
        epochs = 150
        # debugging
        # for i, loss in enumerate(self.losses):
        #     print(loss) if i % 20 == 0 else None
        plt.plot(range(epochs), self.losses)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()
