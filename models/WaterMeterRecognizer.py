import torch
from torchvision import transforms


class WaterMeterRecognizer(torch.nn.Module):
    def __init__(self,
                 rota_reg_net,
                 object_detect_net,
                 digits_classifier,
                 digits_predictor,
                 rota_times=3,
                 classifier_input_shape=(64, 288)):
        super().__init__()
        self.rota_reg_net = rota_reg_net
        self.object_detect_net = object_detect_net
        self.digits_classifier = digits_classifier
        self.digits_predictor = digits_predictor

        self.rota_times = rota_times
        self.classifier_input_shape = classifier_input_shape

    def forward(self, images, targets=None):
        images = list(images)
        if self.training:
            # it's kinda complicated to implement
            # and to train this 3-stage network in end-to-end fashion might be challenging.
            raise NotImplementedError()
        else:
            with torch.no_grad():
                for _ in range(self.rota_times):
                    for idx in range(len(images)):
                        rota_angle = self.rota_reg_net(images[idx])
                        images[idx] = transforms.RandomRotation([rota_angle, rota_angle])(images[idx])
                patches = []
                for img in images:
                    pred = self.object_detect_net(img.unsqueeze(0))[0]['boxes'].to(torch.int)
                    # check if bbox exists
                    bbox = pred[0] if len(pred) != 0 else torch.tensor([0, 0, 1, 1])
                    patch = img[:, bbox[1]: bbox[3], bbox[0]: bbox[2]]
                    patches.append(transforms.Resize(self.classifier_input_shape)(patch))
                predictions = self.digits_predictor(self.digits_classifier(torch.stack(patches, dim=0)))

            return predictions


def water_meter_recognizer_pretrained(rrn_filepath='TrainedModels/RRN.pt',
                                      sodn_file_path='TrainedModels/SODN.pt',
                                      drn_filepath='TrainedModels/DRN.pt'):

    from models.DigitsRecgnitionNetwork import DigitsRecognizer, DigitsPredictor
    from models.RotationRegressionNetwork import RotaRegNetwork, Transform
    from models.SingleObjectDetectionNetwork import SingleObjectDetectionNetwork

    rrn = torch.load(rrn_filepath)
    sodn = torch.load(sodn_file_path)
    drn = torch.load(drn_filepath)

    return WaterMeterRecognizer(rrn, sodn, drn, DigitsPredictor())
