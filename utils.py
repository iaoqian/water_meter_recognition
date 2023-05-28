import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np


class TrainVisualizer:
    """
    record data while training and visualize it
    """
    def __init__(self):
        self.data = {}

    def record(self, **kwargs):
        # pass kwargs(key and value pair) to record it, the new data will be appended to the data series.
        # e.g. Entity.record(**{'DATA_0': data_0, 'DATA_1': data_1})
        for key in kwargs.keys():
            if key in self.data.keys():
                self.data[key].append(kwargs[key])
            else:
                self.data[key] = [kwargs[key]]

    def visualize(self, x_axis_key: str, y_axis_keys: list[list or str], sub_figure_size=(7, 5)):
        """
        args:
            x_axis_key: the key of data you want to plot in x-axis(horizontal-axis), string
                it supports specifying one var in single plot so far, cuz that's all we need under this scenario.
            y_axis_keys: list of keys of vars you want to plot in y-axis(vertical-axis),
                you can GROUP some vars to plot them in one sub-figure for comparison or something.
        example:
            while(training):
                -> do something
                -> visualizer.record({
                    'EPOCH': epoch, 'LOSS': loss, 'Train_Accuracy': train_acc, 'Valid_Accuracy': valid_acc
                })
            -> end training
            -> visualizer.visualize(x_axis_key='EPOCH', y_axis_key=['LOSS', ['Train_Accuracy', 'Valid_Accuracy']])
            # in this way you'll see 2 subplots, which are loss-epoch fig and (train_acc, valid_acc)-epoch.
        """
        x = self.data[x_axis_key]
        sub_plots = len(y_axis_keys)
        plt.figure(figsize=(sub_figure_size[0], sub_figure_size[1] * sub_plots))
        for sub_plot_idx in range(sub_plots):
            plt.subplot(sub_plots, 1, sub_plot_idx + 1)
            if not isinstance(y_axis_keys[sub_plot_idx], list or tuple):
                y_axis_keys[sub_plot_idx] = [y_axis_keys[sub_plot_idx],]
            legend = []
            for curve_idx in range(len(y_axis_keys[sub_plot_idx])):
                plt.plot(x, self.data[y_axis_keys[sub_plot_idx][curve_idx]])
                legend.append(y_axis_keys[sub_plot_idx][curve_idx])
            plt.legend(legend, loc='upper left')
        plt.show()

    def reset(self):
        self.data.clear()


def multi_target_loss(labels, logits):
    # loss function for multiple targets recognition
    # there may be some official implementation of it or something like this, but I did it in my way :)
    celoss = torch.nn.CrossEntropyLoss()
    # briefly, labels with shape (batch, num_targets), logits with shape (batch, num_targets, num_classes)
    # in this project, we tried to recognize 5 targets(i.e. read 5 digits in water meter) and 20 different
    # classes in our definition.
    # we use average loss on 5 targets as final loss, all we need to do is compute losses in every single target
    # to do that, we compute as follows.
    losses = [celoss(lo, la) for lo, la in zip(logits.transpose(0, 1), labels.transpose(0, 1))]
    return sum(losses) / len(losses)


def sum_list_strs(list_strs):
    # [str0, str1, str2, str3, ..., ]
    # concat these strings together
    result = ''
    length = len(list_strs)
    for i in range(length):
        result = result + ' ' + list_strs[i]
    return result[1:]


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
    # draw_bounding_boxes implemented in torchvision supports image input with dtype=uint8 only
    # that's too annoying. so, build this wheel to relief my pain :)
    return draw_bounding_boxes(transforms.PILToTensor()(transforms.ToPILImage()(img)), bbox, labels=label)


def tuple_collate_fn(batch):
    # collate input images with different sizes into a batch
    return tuple(zip(*batch))


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
        # rotation matrix, see in https://en.wikipedia.org/wiki/Rotation_matrix
        R = torch.tensor([[torch.cos(rad), - torch.sin(rad)],
                          [torch.sin(rad), torch.cos(rad)]])
        Rv = R @ v
        Rv += torch.tensor([X, Y])
        return Rv

    new_coors = []
    # default: x <--> W, y <--> H
    # if you want to swap them, enable it.
    if swap_xy:
        for i in range(len(coordinates) // 2):
            new_coors.append(rotate_one_coordinate(coordinates[2 * i + 1], coordinates[2 * i]))
    else:
        for i in range(len(coordinates) // 2):
            new_coors.append(rotate_one_coordinate(coordinates[2 * i + 1], coordinates[2 * i]).flip(dims=(0,)))

    return torch.cat(new_coors, dim=0)