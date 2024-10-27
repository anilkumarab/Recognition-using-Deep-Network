# Arun Srinivasan V K (002839500), and Abhinav Anil (002889398)
# 4/2/2024
# Purpose - This is the file for task_1 A, B, C, D

import sys
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import NN_Model
import matplotlib
matplotlib.use('TkAgg')


# train_loop - for training the model (used onehot encoding - not necessary, but we thought why not
def train_loop(data, model, optim, loss, attributes, train_error_list, test_error_list, test_error, train_acc_list, test_acc_list, test_acc) :
    """
    :param data:
    :param model:
    :param optim:
    :param loss:
    :param attributes:
    :param train_error_list:
    :param test_error_list:
    :param test_error:
    :param train_acc_list:
    :param test_acc_list:
    :param test_acc:
    """
    model.train()
    correct = 0
    count = 0

    for batch, (img, target) in enumerate(data) :
        img = img.to(attributes.device)
        target = target.to(attributes.device)
        output = model(img)
        loss_cal = loss(output, target)

        loss_cal.backward()
        optim.step()
        optim.zero_grad()

        fin_output = []
        for out in output:
            out = out.cpu().detach().numpy()
            out_max = max(out)
            out = (out == out_max).astype('int')
            fin_output.append(out)

        fin_output = np.array(fin_output)
        fin_output = torch.tensor(fin_output).to(attributes.device)
        correct += (fin_output == target).type(torch.float).sum().item()

        count += len(output)

        train_error_list.append(loss_cal.item())
        test_error_list.append(test_error)
        train_acc_list.append(10 * correct/count)
        test_acc_list.append(test_acc)

        if batch % 100 == 0: # to keep track of the loss during training
            loss_val, current = loss_cal.item(), batch * attributes.batch_size + len(img)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{len(data.dataset):>5d}]")


# test_loop is for testing the model during every epoch - to check how the model is learning/ performing
def test_loop(data, model, loss, attributes) :
    """
    :param data:
    :param model:
    :param loss:
    :param attributes:
    :return:
    """
    model.eval()
    correct = 0
    loss_cal = 0
    batch_size = len(data)

    with torch.no_grad() :
        for img, target in data :
            img = img.to(attributes.device)
            output = model(img)
            target = target.to(attributes.device)
            loss_cal += loss(output, target).item()
            fin_output = []
            for out in output :
                out = out.cpu().detach().numpy()
                out_max = max(out)
                out = (out == out_max).astype('int')
                fin_output.append(out)

            fin_output = np.array(fin_output)
            fin_output = torch.tensor(fin_output).to(attributes.device)
            correct += (fin_output == target).type(torch.float).sum().item()

        loss_cal /= batch_size
        correct /= len(data.dataset) * attributes.classes # since we're doing one-hot encoding we've to multiply with the no. of classes
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss_cal:>8f} \n")

    return loss_cal, 100 * correct


def main(argv) :
    # we actually stored all the non-changing data in NN_Model. Btw NN_Model has network-archi in it (task_1 B)
    attributes = NN_Model.Base_attributes()

    train_data_pre = torchvision.datasets.MNIST(root="datasets", train=True, transform=attributes.transforms,
                                                target_transform=attributes.target_transforms, download=True)
    test_data_pre = torchvision.datasets.MNIST(root="datasets", train=False, transform=attributes.transforms,
                                               target_transform=attributes.target_transforms, download=True)

    model = NN_Model.create_model().to(attributes.device)

    print(f"device status {attributes.device}\n\n\n")

    train_data = DataLoader(train_data_pre, batch_size=attributes.batch_size)
    test_data = DataLoader(test_data_pre, batch_size=attributes.batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=attributes.lr)

    # plotting some images in the training set - task_1 A
    fig, axs = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            index = i * 3 + j
            axs[i, j].imshow(train_data_pre.data[index], cmap='gray')
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()

    train_error_list = []
    test_error_list = []
    train_acc_list = []
    test_acc_list = []
    test_error = 3.0
    test_acc = 0

    for epoch in range(attributes.no_of_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_data, model, optimizer, criterion, attributes, train_error_list, test_error_list, test_error, train_acc_list, test_acc_list, test_acc)
        test_error, test_acc = test_loop(test_data, model, criterion, attributes)

    # NOTE - in both plots we actually matched the test_data same as train_data, because the discrete look of the test_data doesn't look good

    # task_1 C - train and test error plot
    plt.plot(train_error_list)
    plt.plot(test_error_list)
    plt.legend(["train_error", "test_error"])
    plt.xlabel("batch iterations")
    plt.ylabel("error")
    plt.show()

    # task_1 C - train and test accuracy plot
    plt.plot(train_acc_list)
    plt.plot(test_acc_list)
    plt.legend(["train_acc", "test_acc"])
    plt.xlabel("batch iterations")
    plt.ylabel("% accuracy")
    plt.ylim(80, 110)
    plt.show()

    # task_1 D - save the network/ model
    torch.save(model.state_dict(), attributes.model_path)

    return


if __name__ == "__main__" :
    main(sys.argv)
