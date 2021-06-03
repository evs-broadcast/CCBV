#!/usr/bin/env python


###########
# Imports #
###########

import os
import json
from pathlib import Path

import torch
import torchvision.transforms as transforms

from kornia.geometry.warp import HomographyWarper, normalize_homography
from sklearn.metrics import jaccard_score
from PIL import Image
from tqdm import tqdm

from dictionary import Dictionary
from field import Field
from models import UNet, Siamese, STN


#############
# Functions #
#############

def four2three(img):
    """
    Convert an image represented by a 4 * h * w tensor
    to a 3 * h * w tensor.
    """

    if len(img.size()) > 3:
        img = img.squeeze(0)

    h, w = img.size()[1], img.size()[2]
    converted = torch.zeros(3, h, w)

    colors = torch.argmax(img, dim=0)

    converted[0][(colors == 3).squeeze()] = 1
    converted[1][(colors == 2).squeeze()] = 1
    converted[2][(colors == 1).squeeze()] = 1

    return converted


def export(outputs, path, name, idx):
    img = outputs[0]
    img = four2three(img)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(path, '{}-{}.png'.format(name, idx)))


def evaluate_warp(outputs, targets, maximize=True):
    if maximize:
        outputs = torch.argmax(outputs, dim=1)
    else:
        outputs = torch.argmax(outputs, dim=1)
        targets = torch.argmax(targets, dim=1)
    ious = []
    for i in range(outputs.size(0)):
        o = torch.flatten(outputs[i])
        t = torch.flatten(targets[i])

        iou = jaccard_score(t, o, average='weighted')
        ious.append(iou)

    return ious


class HomographyPredictor:

    def __init__(self,
                 weights='model_weights/',
                 dictionary='calibration_data/dictionary.json',
                 field='calibration_data/model.png'
                 ):
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # Models
        self.models = [UNet(3, 4), Siamese(4, 128), STN(8, 8)]
        self.models = [model.to(self.device) for model in self.models]

        weights = [os.path.join(weights, model.__class__.__name__.lower() + '.pth') for model in self.models]

        for model, weight in zip(self.models, weights):
            model.load_state_dict(torch.load(weight, map_location=self.device))

        for model in self.models:
            model.eval()

        # Ground truth field model
        self.field = Field(field)
        self.transform = transforms.ToTensor()

        # Dictionary
        print('Loading dictionary...')
        self.dictionary = Dictionary(dictionary, self.field, from_scratch=True)

        # Warper
        self.warper = HomographyWarper(144, 256, mode='nearest')

    def predict_homography(self, image, outpt=None, idx=None):
        sx = image.size[0] / 256
        sy = image.size[1] / 144
        scaling_homography = torch.FloatTensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

        img = image.resize((256, 144), Image.LANCZOS)
        img = self.transform(img)
        inputs = img.to(self.device)
        inputs = inputs.unsqueeze(0)

        with torch.no_grad():
            # Segmentation
            masks = self.models[0](inputs)
            colors = torch.max(masks, 1, keepdim=True)[0]
            colors = colors.expand(colors.size()[0], 4, colors.size()[2], colors.size()[3])
            processed_masks = colors == masks
            processed_masks[:, 0, :, :] = 0
            processed_masks = processed_masks.float().to(self.device)

            # Template matching
            matches, ht, _ = self.dictionary.match(processed_masks, self.models[1], self.device)
            matches = matches.to(self.device)
            ht = ht.to(self.device)

            # Homography refinement
            x = torch.cat((processed_masks, matches), dim=1)
            theta = self.models[2](x)

        ones = torch.ones(x.size()[0], 1).to(self.device)
        theta = torch.cat((theta, ones), dim=1)

        ht, theta = ht.view(-1, 3, 3), theta.view(-1, 3, 3)

        # Get the final homography of the input
        homographies = torch.matmul(ht, theta)

        # Fields warping
        fhomographies = torch.matmul(self.field.h.to(self.device), torch.inverse(homographies))
        fhomographies = normalize_homography(fhomographies, (144, 256), (144, 256))

        fields = self.field.four.unsqueeze(0).repeat(x.size()[0], 1, 1, 1).to(self.device)

        w_fields = self.warper(fields.float(), fhomographies.float())
        if outpt is not None and idx is not None:
            # Masks exportation...
            export(masks.cpu(), outpt, 'masks', idx)

            # Template matched exportation...
            export(matches.cpu(), outpt, 'matches', idx)

            # Warped images exportation...
            export(w_fields.cpu(), outpt, 'warped-field', idx)

        confidence = evaluate_warp(processed_masks.cpu(), w_fields.cpu(), maximize=False)

        homographies = homographies.cpu()
        result = []
        for i in range(len(confidence)):
            scaled_homography = torch.matmul(scaling_homography, homographies[i])
            result.append(
                {
                    "homography": scaled_homography.flatten().numpy().tolist(),
                    "confidence": confidence[i].item()
                }
            )
        return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Get a predicted output of the model')

    parser.add_argument('-i', '--input', type=str,  help='path to input folder')
    parser.add_argument('-o', '--output', type=str, default='outputs/', help='path to outputs file')
    parser.add_argument('-s', '--save_masks', type=str,  help='path to outputs folder')

    args = parser.parse_args()

    input_images = []
    if os.path.isdir(args.input):
        for img_path in os.listdir(args.input):
            if ".png" in img_path or ".jpg" in img_path:
                input_images.append(img_path)
    print('Number of images to process: {}'.format(len(input_images)))

    # Masks folder
    if args.save_masks is not None :
        if not os.path.isdir(args.save_masks):
            os.makedirs(args.save_masks)

    predictor = HomographyPredictor()
    results = {}

    for image_path in tqdm(input_images):
        image = Image.open(os.path.join(args.input, image_path))
        image_filename = Path(image_path).stem
        homography = predictor.predict_homography(image, args.save_masks, image_filename)
        results[image_path] = homography
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)

