# Arun Srinivasan V K (002839500), and Abhinav Anil (002889398)
# 4/2/2024
# Purpose - This file is for saving network/ model architecture and some parameters

import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# model archi.
class DNNModel(torch.nn.Module) :
    def __init__(self, conv_lyr, conv_kernel, pool_kernel, fc_lyr, drop) :
        super().__init__()

        self.conv0 = self._conv_layer(1, conv_lyr, conv_kernel)
        self.conv1 = self._conv_layer(conv_lyr, 20, conv_kernel)

        self.pool0 = self._pool_layer(2)
        self.pool1 = self._pool_layer(pool_kernel)

        self.activation0 = torch.nn.ReLU()
        self.activation1 = torch.nn.ReLU()
        self.activation2 = torch.nn.ReLU()

        self.flatten = torch.nn.Flatten()

        self.dropout0 = torch.nn.Dropout(drop)

        self.inp_layer = 28 - conv_kernel + 1
        self.inp_layer = int(self.inp_layer/2)
        self.inp_layer = self.inp_layer - conv_kernel + 1
        self.inp_layer = int(self.inp_layer/pool_kernel)
        self.inp_layer = self.inp_layer * self.inp_layer * 20

        self.fc0 = self._fc_layer(self.inp_layer, fc_lyr)
        self.fc1 = self._fc_layer(fc_lyr, 10)

    def _conv_layer(self, in_channel, out_channel, k_size):
        return torch.nn.Conv2d(in_channel, out_channel, k_size)

    def _pool_layer(self, k_size):
        return torch.nn.MaxPool2d(k_size)

    def _fc_layer(self, in_feature, out_feature):
        return torch.nn.Linear(in_feature, out_feature)

    def forward(self, x):
        x = self.conv0(x)
        x = self.pool0(x)
        x = self.activation0(x)
        x = self.conv1(x)
        x = self.dropout0(x)

        x = self.pool1(x)
        x = self.activation1(x)
        x = self.flatten(x)
        x = self.fc0(x)
        x = self.activation2(x)
        x = self.fc1(x)

        return x


# for creating the model with default parameters
def create_model():
    return DNNModel(10, 5, 2, 50, 0.5)


# class containing the base parameters, almost all the .py files use 'em
class Base_attributes:
    def __init__(self) :
        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
            ]
        )
        self.target_transforms = torchvision.transforms.Lambda(
            lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = 1e-2
        self.batch_size = 32
        self.no_of_epochs = 9
        self.classes = 10
        self.model_path = "/home/arun/Downloads/MNIST_model_wght.pth"
        self.greek_model_path = "/home/arun/Downloads/greek_model_wght.pth"
        self.greek_data = "/home/arun/Downloads/greek_train/greek_train"


# this function is for plotting the images - used in analyze.py, extensions.py, test_manual_data.py
def plot(nrows, ncols, img_lst, txt, label, cmap) :
    fig, axs = plt.subplots(nrows, ncols)

    for i in range(nrows):
        for j in range(ncols):
            index = i*ncols + j

            if len(img_lst) <= index :
                axs[i, j].axis('off')
                continue

            axs[i, j].imshow(img_lst[index], cmap=cmap)
            axs[i, j].set_title(f"{label} {txt[index]}")
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()


'''
import torch
import torchsummary
from project_5 import NN_Model

model = NN_Model.DNNModel(12, 7, 3, 20, 0.5)
model.to("cuda")
torchsummary.summary(model, (1, 28, 28))

'''

