"""
Implementation of the ground truth field
manager.
"""

###########
# Imports #
###########

import torch
import torchvision.transforms as transforms
from kornia.geometry.transform import warp_perspective

from PIL import Image


#####################
# General variables #
#####################

# Matrices associated to the ground truth model
CALIBRATION = torch.tensor([
    [150, 0, 960],
    [0, 150, 540],
    [0, 0, 1]
]).float()

HOMOGRAPHY = torch.tensor([
    [15, 0, 960],
    [0, 15, 540],
    [0, 0, 1]
]).float()


###########
# Classes #
###########
def three2four(img):
    """
    Convert an image represented by a 3 * h * w tensor
    to a 4 * h * w tensor.
    """

    if len(img.size()) > 3:
        img = img.squeeze(0)

    h, w = img.size()[1], img.size()[2]
    converted = torch.zeros(4, h, w)
    nonnul = torch.sum(img, dim=0)
    colors = torch.argmax(img, dim=0)

    colors[(nonnul == 0)] = 4
    converted[1][(colors == 2).squeeze()] = 1
    converted[2][(colors == 1).squeeze()] = 1
    converted[3][(colors == 0).squeeze()] = 1
    return converted


class Field:
    def __init__(self, field, ratio=7.5):
        # Open image file
        self.four = Image.open(field)

        # Transform images to tensors
        self.four = transforms.ToTensor()(self.four)
        self.four = three2four(self.four)
        # Homography to warp (resized) model
        kprim = torch.true_divide(CALIBRATION, ratio)
        kprim[2, 2] = 1

        self.h = torch.matmul(kprim, torch.inverse(CALIBRATION))
        self.h = torch.matmul(self.h, HOMOGRAPHY)

    def warp_field(self, homography):

        hr = torch.matmul(HOMOGRAPHY, torch.inverse(homography.view(3,3)))
        hr = torch.inverse(hr)
        hr = hr.unsqueeze(0)

        # Warp the model
        warped_field = warp_perspective(self.four.unsqueeze(0), hr, (144, 256), flags='nearest')
        return warped_field
