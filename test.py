# Arun Srinivasan V K (002839500), and Abhinav Anil (002889398)
# 4/2/2024
# Purpose - This file is for task_1 E

import sys
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import NN_Model
import matplotlib
matplotlib.use('TkAgg')


def test_loop(data, model, loss, attributes) :
    """
    :param data:
    :param model:
    :param loss:
    :param attributes:
    :return:
    """

    with torch.no_grad() :
        img, target = data
        img = img.unsqueeze(1)
        img = img.to(attributes.device)
        output = model(img)
        target = target.to(attributes.device)

        out = output.cpu().detach().numpy()
        out_max = max(out[0])
        fin_output = (out == out_max).astype('int')

        print("10 network outputs : ", end= "\t")
        for i in out :
            val = np.round(i, 2)
            print(val, end="   ")
        print()
        print(f"Target_index : {target.argmax()}")
        print(f"Predicted_index : {fin_output.argmax()}", "\n")

        return fin_output.argmax()


def main(argv) :
    attributes = NN_Model.Base_attributes()

    test_data_pre = torchvision.datasets.MNIST(root="datasets", train=False, transform=attributes.transforms,
                                               target_transform=attributes.target_transforms, download=True)

    model = NN_Model.create_model().to(attributes.device)
    model.load_state_dict(torch.load(attributes.model_path))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    fig, axs = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            index = i*3 + j
            print(f"Example {index + 1}\n-------------------------------")

            pred = test_loop(test_data_pre[index], model, criterion, attributes)
            axs[i, j].imshow(test_data_pre.data[index], cmap='gray')
            axs[i, j].set_title(f"Prediction : {pred}")
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()

    return


if __name__ == "__main__" :
    main(sys.argv)
