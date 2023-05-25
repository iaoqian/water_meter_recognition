import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np


def multi_target_loss(labels, logits):
    celoss = torch.nn.CrossEntropyLoss()
    return sum([celoss(lo, la) for lo, la in zip(logits.transpose(0, 1), labels.transpose(0, 1))]) / 5


def sum_list_strs(list_strs):
    result = ''
    length = len(list_strs)
    for _ in range(length):
        if _ != 0:
            result += ' '
        result += list_strs[_]
    return result


def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def draw_bbox_with_tensor(img: torch.Tensor, bbox: torch.Tensor, label=None):
    return draw_bounding_boxes(transforms.PILToTensor()(transforms.ToPILImage()(img)), bbox, labels=label)


def tuple_collate_fn(batch):
    return list(zip(*batch))


def object_detection_transform(target):
    # [x0, y0, x1, y1, .. , ]
    coordinate_list = [int(str_digit) for str_digit in target.split()[0: 8]]
    # bbox, format: X_lt, Y_lt, X_rb, Y_rb
    single_bbox = torch.tensor([[coordinate_list[0], coordinate_list[1], coordinate_list[4], coordinate_list[5]]])
    target = {'boxes': single_bbox, 'labels': torch.tensor([1])}
    return target


def coordinate_rotation_transform(coordinates: list[int],
                                  rotation_center_point: tuple[int],
                                  degrees: int,
                                  swap_xy=False,
                                  ):
    # input coordinates format (x0, y0, x1, y1,..,), center point format (X, Y)
    X, Y = rotation_center_point[0], rotation_center_point[1]

    def rotate_one_coordinate(x, y):
        v = torch.tensor([x - X, y - Y]).to(torch.float)
        rad = torch.tensor([degrees / 180]) * torch.pi
        R = torch.tensor([[torch.cos(rad), - torch.sin(rad)],
                          [torch.sin(rad), torch.cos(rad)]])
        Rv = R @ v
        Rv += torch.tensor([X, Y])
        return Rv

    new_coors = []
    # default: x <--> H, y <--> W
    if swap_xy:
        for i in range(len(coordinates) // 2):
            new_coors.append(rotate_one_coordinate(coordinates[2 * i + 1], coordinates[2 * i]))
    else:
        for i in range(len(coordinates) // 2):
            new_coors.append(rotate_one_coordinate(coordinates[2 * i + 1], coordinates[2 * i]).flip(dims=(0,)))

    return torch.cat(new_coors, dim=0)



