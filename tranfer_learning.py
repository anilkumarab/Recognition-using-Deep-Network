# Arun Srinivasan V K (002839500), and Abhinav Anil (002889398)
# 4/2/2024
# Purpose - This file is for performing task_3 - transfer learning on greek letters

import sys
import torch
import torchvision
import NN_Model
from torchvision.transforms import functional
from torch.utils import data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def train_loop(data, model, optim, loss, attributes, train_error_list, batch_size_temp) :
    """
    :param data:
    :param model:
    :param optim:
    :param loss:
    :param attributes:
    :param train_error_list:
    :param batch_size_temp:
    """
    model.train()
    correct = 0
    loss_cal = 0
    batch_size = len(data)
    count = 0

    for batch, (img, target) in enumerate(data) :
        img = img.to(attributes.device)
        target = target.to(attributes.device)
        output = model(img)
        loss_cal_temp = loss(output, target)
        loss_cal += loss_cal_temp.item()

        loss_cal_temp.backward()
        optim.step()
        optim.zero_grad()

        train_error_list.append(loss_cal_temp.item())
        correct += (output.argmax(1) == target).type(torch.float).sum().item()
        count += len(output)

        loss_val, current = loss_cal_temp.item(), batch * batch_size_temp + len(img)
        print(f"loss: {loss_val:>7f}  [{current:>2d}/{len(data.dataset):>2d}]")
        print(f"Test Error: \nAccuracy: {(100 * correct/count):>0.1f}%, Avg loss: {loss_cal/batch_size:>8f} \n")


# prof's code :)
class GreekTransform :
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        x = torchvision.transforms.functional.invert(x)
        return x


# parameters are set as said by the prof
def main(argv) :
    attributes = NN_Model.Base_attributes()
    model = NN_Model.create_model().to(attributes.device)
    model.load_state_dict(torch.load(attributes.model_path))

    model.fc1 = torch.nn.Linear(50, 3)

    for param in model.parameters() :
        param.requires_grad = False

    for param in model.fc1.parameters() :
        param.requires_grad = True

    model.to(attributes.device)

    # prof's code :)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            GreekTransform(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    batch_size_temp = 5
    train_data_pre = torchvision.datasets.ImageFolder(attributes.greek_data, transform)
    train_data = torch.utils.data.DataLoader(train_data_pre, batch_size_temp, True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params= model.parameters(), lr= attributes.lr)

    train_error_list = []
    epochs = 20

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_data, model, optimizer, criterion, attributes, train_error_list, batch_size_temp)

    plt.plot(train_error_list)
    plt.legend(["train_error"])
    plt.xlabel("batch iterations")
    plt.ylabel("error")
    plt.show()

    torch.save(model.state_dict(), attributes.greek_model_path)


if __name__ == "__main__" :
    main(sys.argv)