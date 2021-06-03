"""
Implementation of the dictionary that manages
the different templates.
"""

###########
# Imports #
###########

import io
import json
import torch
import torchvision.transforms as transforms

from PIL import Image
from random import randint
from tqdm import tqdm


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
###########
# Classes #
###########

class Dictionary:
    """
    The dictionary manages all the templates.
    """

    def __init__(self, json_dict, field=None, from_scratch=False):
        # Templates and associated homographies
        self.pairs = []

        # Pre computed elements (data matches and templates encoded)
        self.d_matches, self.t_encoded = {}, []

        # Save each template and associated homography
        with open(json_dict) as json_file:
            data = json.load(json_file)
            count = 0
            for template in tqdm(data):
                if not from_scratch:
                    with open(template['image'].replace('dictionary-four', 'dictionary-three'), 'rb') as img:
                        self.pairs.append((
                            self.transform(io.BytesIO(img.read())),
                            torch.tensor([template['homography_resize']])
                        ))
                else:
                    homography = torch.tensor([template['homography_resize']])
                    template = field.warp_field(homography)
                    self.pairs.append((
                        self.transform2(template),
                        homography
                    ))
                    count += 1

    def pick(self, similarities):
        # Pick a template and associate similarity for each
        # image in the batch
        batch_size = similarities.size()[0]
        templates, y = [], []

        for i in range(batch_size):
            # Decide if we pick a similar or dissimilar
            choice = randint(0, 1)
            if choice:
                idx = similarities[i].item()
            else:
                idx = randint(0, len(self.pairs) - 1)
                if idx == similarities[i].item():
                    idx -= 1
                    if idx < 0:
                        idx += 2

            template, _ = self.pairs[idx]
            # template = self.transform(template)

            similarity = choice

            templates.append(template)
            y.append(similarity)

        templates = torch.stack(templates, dim=0)
        y = torch.tensor(y)
        y = y.view(batch_size, 1)

        return templates, y

    def pick_force(self, similarities, force=1):
        batch_size = similarities.size()[0]
        templates, homographies = [], []

        for i in range(batch_size):
            choice = force

            if choice:
                idx = similarities[i].item()
            else:
                idx = randint(0, len(self.pairs) - 1)
                if idx == similarities[i].item():
                    idx -= 1
                    if idx < 0:
                        idx += 2

            template, h = self.pairs[idx]
            # template = self.transform(template)

            templates.append(template)
            homographies.append(h)

        templates = torch.stack(templates, dim=0)
        homographies = torch.stack(homographies, dim=0)

        return templates, homographies

    def encode(self, model):
        print("Encoding dictionary")
        model = model.to('cpu')

        for pair in self.pairs:
            # template = self.transform(pair[0])
            template = pair[0]
            template = template.unsqueeze(0)
            template = model.encode(template)

            self.t_encoded.append(template)
        print("Encoding done !")

    def match(self, masks, model, device, names=None):
        batch_size = masks.size()[0]
        matches, homographies, idxs = [], [], []

        # Encode dictionary templates if not done
        if len(self.t_encoded) == 0:
            self.encode(model)

        model = model.to(device)

        # Calculate distances between masks and each template and
        # get the matching template
        for i in range(batch_size):
            closest = None if names is None else self.d_matches.get(names[i])

            if closest is None:
                dist = []

                encoded = model.encode(masks[i].unsqueeze(0))

                # for template in self.t_encoded:
                #     template = template.to(device)
                #     dist.append(F.pairwise_distance(encoded, template).item())
                # # print(dist.size())
                # closest = dist.index(min(dist))
                templates = torch.stack(self.t_encoded)
                templates = templates.to(device)
                templates = templates.squeeze(1)

                d = templates - encoded
                dist = torch.norm(d, dim=1, p=None)
                closest = dist.topk(3, largest=False)
                closest = closest.indices[0]

                if names is not None:
                    self.d_matches[names[i]] = closest

            idxs.append(closest)

            match, homography = self.pairs[closest]
            # match = self.transform(match)

            matches.append(match)
            homographies.append(homography)

        matches = torch.stack(matches, dim=0)
        homographies = torch.stack(homographies, dim=0)

        return matches, homographies, idxs



    def transform(self, template):
        template = Image.open(template)
        template = transforms.ToTensor()(template)
        template = template.mul(255.)
        template = three2four(template)

        return template

    def transform2(self, template):
        template = template.squeeze(0)

        return template
