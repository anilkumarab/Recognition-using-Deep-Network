# Arun Srinivasan V K (002839500), and Abhinav Anil (002889398)
# 4/2/2024
# Purpose - This is the file for task_1 F, a part of task_3

import os
import sys
import cv2
import torch
import numpy as np
import torchvision
from torchvision.transforms import functional
import NN_Model


class Manual_transform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.invert(x)
        return x


# prof's code :)
class Greek_transform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        x = torchvision.transforms.functional.invert(x)
        return x


def test_loop(data, model, attributes) :
    with torch.no_grad() :
        img = data
        img = img.unsqueeze(1)
        img = img.to(attributes.device)
        output = model(img)

        out = output.cpu().detach().numpy()
        out_max = max(out[0])
        fin_output = (out == out_max).astype('int')

        print("10 network outputs : ", end="\t")
        for i in out:
            val = np.round(i, 2)
            print(val, end="   ")
        print()
        print(f"Predicted_index : {fin_output.argmax()}", "\n")

        return fin_output.argmax()


def main(argv) :
    attributes = NN_Model.Base_attributes()

    user_input = input("Test with custom data - digits(d)/ greek letters(g)")
    # d is for task_1 F
    # g is a part of task_3

    model = NN_Model.create_model().to(attributes.device)

    img_list = []
    dataloader = []
    txt_list = []

    if user_input == "d":
        PATH = "/home/arun/Downloads/digits"
        model.load_state_dict(torch.load(attributes.model_path))
        nrows = 5
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                Manual_transform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    elif user_input == "g":
        PATH = "/home/arun/Downloads/greek"
        model.fc1 = torch.nn.Linear(50, 3)
        model.to(attributes.device)
        nrows = 3
        model.load_state_dict(torch.load(attributes.greek_model_path))
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                Greek_transform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    else :
        print("Invalid expression...")
        return

    model.eval()

    for pth in os.listdir(PATH) :
        img = cv2.imread(PATH + "/" + pth)
        img_list.append(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #cv2.threshold(img, 130, 255, cv2.THRESH_BINARY, img)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 7)
        temp = transforms(img)
        dataloader.append(temp)
        txt_list.append(test_loop(temp, model, attributes))
        #img_list.append(temp.permute(1, 2, 0).numpy())

    NN_Model.plot(nrows, 2, img_list, txt_list, "Prediction: ", "gray")

    return


if __name__ == "__main__" :
    main(sys.argv)
