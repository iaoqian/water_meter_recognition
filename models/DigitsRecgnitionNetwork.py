import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import PIL.Image


class SubSampleBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, padding='same')
        self.conv1 = torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, padding='same')
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = torch.nn.ReLU()
        self.bn0 = torch.nn.BatchNorm2d(out_dim)
        self.bn1 = torch.nn.BatchNorm2d(out_dim)

        self.conv1x1 = torch.nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, X):
        return self.conv1x1(self.max_pool(X)) + \
            self.bn1(self.relu(self.max_pool(self.conv1(self.bn0(self.relu(self.conv0(X)))))))


class ResidualBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, X):
        return X + self.module(X)


class MaintainBlock(torch.nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            seq = torch.nn.Sequential(
                torch.nn.Conv2d(dim, dim, kernel_size=3, padding='same'),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(dim),
            )
            layers.append(ResidualBlock(seq))
        self.forward_blk = torch.nn.Sequential(*layers)

    def forward(self, X):
        return self.forward_blk(X)


class DigitsRecognizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # shape [batch, 3, 64, 288]
        self.sb0 = SubSampleBlock(3, 64)
        # [..., 64, 32, 144]
        self.sb1 = SubSampleBlock(64, 256)
        # [..., 256, 16, 72]
        self.mb0 = MaintainBlock(256, 5)
        # [..., 256, 16, 72]
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 20, kernel_size=(4, 8), stride=4, padding=(0, 4)),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(320, 100),
            torch.nn.Unflatten(dim=1, unflattened_size=(5, 20))
        )

    def forward(self, X):
        return self.head(self.mb0(self.sb1(self.sb0(X))))


class DigitsPredictor:
    int2str_labels = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: '01', 11: '12', 12: '23', 13: '34', 14: '45', 15: '56', 16: '67', 17: '78', 18: '89', 19: '90'
    }

    @staticmethod
    def predict(logits):
        batch_preds = []
        logits = logits.cpu().detach()
        logits = logits.argmax(dim=2)
        for row in logits:
            pred = []
            preds = []
            idx = []
            for i in range(5):
                val = DigitsPredictor.int2str_labels[row[i].item()]
                pred.append(val)
                idx.append(len(val))

            for a in range(idx[0]):
                for b in range(idx[1]):
                    for c in range(idx[2]):
                        for d in range(idx[3]):
                            for e in range(idx[4]):
                                preds.append(pred[0][a] + pred[1][b] + pred[2][c] + pred[3][d] + pred[4][e])
            batch_preds.append(preds)
        return batch_preds

    def __call__(self, arg):
        return self.predict(arg)