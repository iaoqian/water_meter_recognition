import torch
import torchvision
import torchvision.transforms as transforms
import PIL.Image


class Transform(torch.nn.Module):
    def __init__(self, img_transforms=None):
        super().__init__()
        self.origin_sizes = []

        if img_transforms is not None:
            self.img_transforms = img_transforms
        else:
            self.img_transforms = self._default_transforms

    def _default_transforms(self, image, angels=None):
        # input image with size (W, H)
        if isinstance(image, PIL.Image.Image):
            image = transforms.ToTensor()(image)

        origin_size = image.shape[1: 3]

        self.origin_sizes.append(origin_size)
        if origin_size[0] / origin_size[1] >= 1.5:
            image = transforms.Resize((480, 270))(image)
            if angels:
                angels = torch.atan(
                    torch.tan(angels * 90 / 180 * torch.pi) / (origin_size[0] / 480.0) * (origin_size[1] / 270.0)
                )
        elif origin_size[0] / origin_size[1] <= 0.67:
            image = transforms.Resize((270, 480))(image)
            if angels:
                angels = torch.atan(
                    torch.tan(angels * 90 / 180 * torch.pi) / (origin_size[0] / 270.0) * (origin_size[1] / 480.0)
                )
        else:
            image = transforms.Resize((270, 270))(image)
            if angels:
                angels = torch.atan(
                    torch.tan(angels * 90 / 180 * torch.pi) / (origin_size[0] / 270.0) * (origin_size[1] / 270.0)
                )

        if angels is not None:
            return transforms.CenterCrop((224, 224))(image), angels / torch.pi * 180.0 / 90.0
        else:
            return transforms.CenterCrop((224, 224))(image)

    def forward(self, images: torch.Tensor or tuple or list, angels=None):
        if isinstance(images, torch.Tensor):
            images = [images]
        if isinstance(images, tuple):
            images = list(images)
        if isinstance(angels, torch.Tensor):
            angels = [angels]
        if isinstance(angels, tuple):
            angels = list(angels)
        # origin_sizes could be non-empty due to unexpected interrupt or other reasons, so clear it.
        self.origin_sizes.clear()
        if angels is not None:
            for i in range(len(images)):
                images[i], angels[i] = self.img_transforms(images[i], angels[i])
        else:
            for i in range(len(images)):
                images[i] = self.img_transforms(images[i])
        # convert to 4-D things
        if angels is not None:
            return torch.stack(images, dim=0), torch.cat(angels, dim=0)
        else:
            return torch.stack(images, dim=0), None

    def post_process(self, predictions: torch.Tensor):
        # model's outputs are normalized angles(in degree)
        # this method returns real rotation angles in origin images
        predictions = torch.tan(predictions * 90.0 / 180.0 * torch.pi)

        for i in range(predictions.shape[0]):
            origin_size = self.origin_sizes[i]

            if origin_size[0] / origin_size[1] >= 1.5:
                predictions[i] = predictions[i] * (origin_size[0] / 480.0) / (origin_size[1] / 270.0)
            elif origin_size[0] / origin_size[1] <= 0.67:
                predictions[i] = predictions[i] * (origin_size[0] / 270.0) / (origin_size[1] / 480.0)
            else:
                predictions[i] = predictions[i] * (origin_size[0] / 270.0) / (origin_size[1] / 270.0)

        self.origin_sizes.clear()
        return torch.atan(predictions) / torch.pi * 180.0


class RotaRegNetwork(torch.nn.Module):
    # Rotation Angle Regression Detection
    def __init__(self, transform, backbone, output_dim):
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        # todo: reg_head was a improper design, remove it.
        self.reg_head = torch.nn.Linear(output_dim, 1)

    @staticmethod
    def compute_loss(y_hat, y):
        # compute losses
        # you can choose one of them or some of them as your loss value.
        # but I recommend using MAE loss, out of my experience in practice :)
        mse_loss = torch.nn.MSELoss()(y_hat, y)
        mae_loss = torch.nn.L1Loss()(y_hat, y)
        smooth_l1_loss = torch.nn.SmoothL1Loss()(y_hat, y)
        return {'mse_loss': mse_loss, 'mae_loss': mae_loss, 'smooth_l1_loss': smooth_l1_loss}

    def forward(self, images, rota_values=None):
        # todo: fix this improper usage
        device = self.reg_head._parameters['bias'].device

        X, y = self.transform(images, rota_values)
        X = X.to(device)
        y = y.to(device) if y is not None else None
        # always set backbone in train mode
        self.backbone.train()
        y_hat = self.reg_head(self.backbone(X))

        if self.training:
            assert y is not None, 'rota_values can\'t be None when model in training mode.'
            return RotaRegNetwork.compute_loss(y_hat, y), self.transform.post_process(y_hat)
        else:
            return self.transform.post_process(y_hat)


def rota_reg_net_resnet18(pretrained_model_filepath=None):
    if not pretrained_model_filepath:
        backbone = torchvision.models.resnet18()
    else:
        backbone = torch.load(pretrained_model_filepath)

    backbone.fc = torch.nn.Identity()
    config = {
        'backbone': backbone,
        'output_dim': 512,
        'transform': Transform()
    }
    return RotaRegNetwork(**config)
