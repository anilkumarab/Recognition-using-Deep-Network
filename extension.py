# Arun Srinivasan V K (002839500), and Abhinav Anil (002889398)
# 4/2/2024
# Purpose - This is an extension (3/3)

import sys
import torchvision
import numpy as np
import NN_Model
import cv2


# here we used a pre-trained network (resnet50) available in the PyTorch package, and replicated task_2
def main(argv) :
    attributes = NN_Model.Base_attributes()
    model = torchvision.models.resnet50(weights= torchvision.models.ResNet50_Weights.DEFAULT).to(attributes.device)
    model.eval()

    img = cv2.imread("/home/arun/Downloads/c661e223ec6f.jpg")

    label = "filter"
    lst = [i for i in range(64)]
    model_conv0 = next(model.named_parameters())
    img_lst_ = model_conv0[1].cpu().detach().numpy().squeeze()
    img_lst_ = np.clip(img_lst_, 0.0, 1.0)
    img_lst = [np.transpose(i_lst, (1, 2, 0)) for i_lst in img_lst_]
    print(np.max(img_lst_))

    NN_Model.plot(4, 4, img_lst, lst, label, "viridis")
    tot_img_lst = []

    for i in range(16) :
        tot_img_lst.append(img_lst[i])
        out_temp = np.zeros_like(img)

        for j in range(3) :
            out_temp[:, :, j] = cv2.filter2D(img[:, :, j], -1, img_lst[i][:, :, j])

        tot_img_lst.append(out_temp)

    label = ""
    lst = ["" for _ in range(20)]
    NN_Model.plot(5, 4, tot_img_lst, lst, label, "gray")


if __name__ == "__main__" :
    main(sys.argv)