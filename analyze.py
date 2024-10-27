# Arun Srinivasan V K (002839500), and Abhinav Anil (002889398)
# 4/2/2024
# Purpose - This file is for task_2

import sys
import torch
import torchvision
import NN_Model
import cv2


def main(argv) :
    attributes = NN_Model.Base_attributes()
    model = NN_Model.create_model().to(attributes.device)
    model.load_state_dict(torch.load(attributes.model_path))
    model.eval()

    train_data_pre = torchvision.datasets.MNIST(root="datasets", train=True, transform=attributes.transforms,
                                                target_transform=attributes.target_transforms, download=True)

    for i in model.named_parameters() :
        print(i[0], "   ", i[1].shape)

    label = "filter"
    lst = [i for i in range(10)]
    model_conv0 = next(model.named_parameters())
    img_lst = model_conv0[1].cpu().detach().numpy().squeeze()

    # task_2 A - analyze the first layer of the network/ model
    NN_Model.plot(3, 4, img_lst, lst, label, "viridis")

    tot_img_lst = []
    img = train_data_pre[0][0].permute(1, 2, 0).numpy()

    for i in range(10) :
        print(img.shape, img_lst[i].shape)
        tot_img_lst.append(img_lst[i])
        out = cv2.filter2D(img.squeeze(), -1, img_lst[i])
        tot_img_lst.append(out)

    label = ""
    lst = ["" for _ in range(20)]
    # task_2 B - effect of filters of an image
    NN_Model.plot(5, 4, tot_img_lst, lst, label, "gray")


if __name__ == "__main__" :
    # plot of the training error, a printout of your modified network, and the results on the additional data are attached in the report
    main(sys.argv)
