import torch
import torchvision


class SingleObjectDetectionNetwork(torch.nn.Module):
    def __init__(self, transform, backbone, reg_head):
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.reg_head = reg_head

    @staticmethod
    def compute_loss(detections, targets):
        sum_diou_loss = 0
        for det_target, target in zip(detections, targets):
            diou = torchvision.ops.diou_loss.distance_box_iou_loss(det_target['boxes'], target['boxes'])
            sum_diou_loss += diou
        return sum_diou_loss / len(detections)

    def forward(self, images, targets):
        device = self.reg_head._parameters['bias'].device

        origin_sizes = []
        for image in images:
            origin_sizes.append((image.shape[-2], image.shape[-1]))

        images, targets = self.transform(images, targets)

        detections = []
        # return images, targets

        boxes = self.reg_head(self.backbone(images.tensors.to(device)))
        for box, (H, W) in zip(boxes, images.image_sizes):
            box = box ** 2
            box[2: 4] = box[2: 4] + box[0: 2]
            box = box * torch.tensor([H, W, H, W], device=box.device)
            d = {'boxes': box.reshape(1, -1), 'labels': torch.tensor([1])}
            detections.append(d)

        detections = self.transform.postprocess(detections, images.image_sizes, origin_sizes)
        if self.training:
            return self.compute_loss(detections, targets)
        return detections


def single_od_network():
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1, 1, 1.0]
    backbone = torchvision.models.resnet34()
    backbone.fc = torch.nn.Identity()

    from torchvision.models.detection.transform import GeneralizedRCNNTransform
    config = {
        'backbone': backbone,
        'transform': GeneralizedRCNNTransform(min_size=224, max_size=224, image_mean=image_mean, image_std=image_std),
        'reg_head': torch.nn.Linear(512, 4)
    }
    return SingleObjectDetectionNetwork(**config)


def fasterrcnn_mobilenet_v3_large_320_fpn():
    pretrained_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    )
    out_channels = pretrained_model.backbone(torch.rand(1, 3, 1, 1))['0'].shape[1]

    anchor_sizes = ((32, 64, 128, 256, 512,),) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    defaults = {
        "min_size": 320,
        "max_size": 640,
        "rpn_pre_nms_top_n_test": 10,
        "rpn_post_nms_top_n_test": 1,
        "rpn_pre_nms_top_n_train": 10,
        "rpn_post_nms_top_n_train": 1,
        "rpn_score_thresh": 0.05,
    }
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    class DummyBackbone(torch.nn.Module):
        # to fool FasterRCNN class
        def __init__(self, out_channels):
            super().__init__()
            self.out_channels = out_channels

    my_model = FasterRCNN(DummyBackbone(out_channels),
                          num_classes=2,
                          rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
                          **defaults
                          )
    # use backbone in pretrained model directly
    my_model.backbone = pretrained_model.backbone
    return my_model