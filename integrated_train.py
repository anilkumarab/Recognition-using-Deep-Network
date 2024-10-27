# Arun Srinivasan V K (002839500), and Abhinav Anil (002889398)
# 4/2/2024
# Purpose - This is a part of task_4

import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import NN_Model
import wandb


# the use of this file is to train the model whenever it gets the hyperparameters and return the test accuracy
def train_loop(data, model, optim, loss, attributes) :
    model.train()

    for batch, (img, target) in enumerate(data) :
        img = img.to(attributes.device)
        target = target.to(attributes.device)
        output = model(img)
        loss_cal = loss(output, target)

        loss_cal.backward()
        optim.step()
        optim.zero_grad()


def test_loop(data, model, loss, attributes) :
    model.eval()
    correct = 0

    with torch.no_grad() :
        for img, target in data :
            img = img.to(attributes.device)
            output = model(img)
            target = target.to(attributes.device)
            fin_output = []
            for out in output :
                out = out.cpu().detach().numpy()
                out_max = max(out)
                out = (out == out_max).astype('int')
                fin_output.append(out)

            fin_output = np.array(fin_output)
            fin_output = torch.tensor(fin_output).to(attributes.device)
            correct += (fin_output == target).type(torch.float).sum().item()

        correct /= len(data.dataset) * attributes.classes

    return 100 * correct


def start_process(config = None) :
    wandb.init(config=config)
    config = wandb.config
    attributes = NN_Model.Base_attributes()

    train_data_pre = torchvision.datasets.FashionMNIST(root="datasets", train=True, transform=attributes.transforms,
                                                       target_transform=attributes.target_transforms, download=True)
    test_data_pre = torchvision.datasets.FashionMNIST(root="datasets", train=False, transform=attributes.transforms,
                                                      target_transform=attributes.target_transforms, download=True)

    model = NN_Model.DNNModel(config.conv_layer, config.conv_kernel, config.pool_kernel, config.fc, config.dropout).to(attributes.device)

    print(f"\n\n\ndevice-status: {attributes.device} \n\n\n")

    train_data = DataLoader(train_data_pre, batch_size=config.batch_size)
    test_data = DataLoader(test_data_pre, batch_size=config.batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=config.lr)

    test_accuracy = 0

    for epoch in range(config.epochs):

        train_loop(train_data, model, optimizer, criterion, attributes)
        test_accuracy = test_loop(test_data, model, criterion, attributes)

        if epoch > 0 and epoch % 10 == 0:
            print(f"Model currently at {100*epoch/config.epochs}%")

    wandb.log({"test_accuracy" : test_accuracy})
