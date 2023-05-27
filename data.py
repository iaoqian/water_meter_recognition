import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import PIL.Image


class DigitsDateset(data.Dataset):
    """
    Load data segmented from origin data.
    """

    str2int_labels = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        '01': 10, '12': 11, '23': 12, '34': 13, '45': 14, '56': 15, '67': 16, '78': 17, '89': 18, '09': 19
    }

    int2str_labels = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: '01', 11: '12', 12: '23', 13: '34', 14: '45', 15: '56', 16: '67', 17: '78', 18: '89', 19: '09'
    }

    def __init__(self, dataset: str, root: str = 'WaterMeterDataset/ProcessedData', img_transform=None,
                 target_transform=None):
        if dataset == 'test':
            self.images_path = root + '/test_imgs_seg'
            self.label_path = None
            self.num_images = 500
        else:
            self.images_path = root + '/train_imgs_seg'
            self.label_path = root + '/train_labels'
            self.num_images = 1000

        self.img_transform = img_transform if img_transform else self._default_img_transform
        self.target_transform = target_transform if target_transform else self._default_target_transform

    @staticmethod
    def _default_img_transform(img: PIL.Image):
        # simple data augmentation
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((64, 288)),
                transforms.ColorJitter(brightness=0.5, hue=0.2),
                transforms.RandomRotation([-5, 5]),
            ]
        )
        return trans(img)

    @staticmethod
    def _default_target_transform(target: str):
        # target data format [x0, y0, x1, y1, x2, y2, x3, y3, [...digit data...]]
        digits = target.split()[8:]
        label = [digits[0][0], digits[0][1], digits[0][2], digits[0][3], digits[0][4]]
        for i in range(len(digits) - 1):
            for j in range(5):
                if digits[i + 1][j] not in label[j]:
                    # assume 2-3, 3-2 are same, though it's not in dataset
                    # the problem is there is no enough data to support training without this operation
                    if label[j] < digits[i + 1][j]:
                        label[j] = label[j] + digits[i + 1][j]
                    else:
                        label[j] = digits[i + 1][j] + label[j]
        for i in range(5):
            label[i] = DigitsDateset.str2int_labels[label[i]]
        return torch.tensor(label)

    def __getitem__(self, index):
        if not self.label_path:
            image = PIL.Image.open(self.images_path + f'/test_seg_{index + 1}.jpg')
            # while test, not use data augment
            return transforms.Resize((64, 288))(transforms.ToTensor()(image))
        else:
            image = PIL.Image.open(self.images_path + f'/train_seg_{index + 1}.jpg')
            label = open(self.label_path + f'/labels/train_{index + 1}.txt').read()

            return self.img_transform(image), self.target_transform(label)

    def __len__(self):
        return self.num_images


class RawWaterMeterDS(data.Dataset):
    """
    Read raw water meter dataset for pre-processing or something
    """

    def __init__(self, dataset='train', root='./WaterMeterDataset', img_transform=None, target_transform=None):
        if dataset == 'train':
            self.images_path = root + '/train_imgs'
            self.label_path = root + '/train_labels'
            self.num_images = 1000
        else:
            self.images_path = root + '/test_imgs/test_imgs'
            self.label_path = None
            self.num_images = 500

        self.img_transform = img_transform if img_transform else self._default_img_transform
        self.target_transform = target_transform if target_transform else self._default_target_transform

    @staticmethod
    def _default_img_transform(img: PIL.Image):
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=0.5, hue=0.2),
            ]
        )
        return trans(img)

    @staticmethod
    def _default_target_transform(target: str):
        # target data format [x0, y0, x1, y1, x2, y2, x3, y3, [...digit data...]]
        coordinate_list = [int(str_digit) for str_digit in target.split()[0: 8]]
        x_0, y_0, x_1, y_1 = coordinate_list[0: 4]
        if (x_1 - x_0) == 0:
            return torch.tensor([[1.0]])
        k = (y_1 - y_0) / (x_1 - x_0)
        angel = torch.atan(torch.tensor([[k]])) / torch.pi * 180.0 / 90.0

        return angel

    def __getitem__(self, index):

        if not self.label_path:
            image = PIL.Image.open(self.images_path + f'/test_{index + 1}.jpg')
            # not use data augment while handling test images
            return transforms.ToTensor()(image)
        else:
            image = PIL.Image.open(self.images_path + f'/train_{index + 1}.jpg')
            label = open(self.label_path + f'/labels/train_{index + 1}.txt').read()

            return self.img_transform(image), self.target_transform(label)

    def __len__(self):
        return self.num_images
