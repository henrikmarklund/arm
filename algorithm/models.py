import torch
import torch.nn as nn
import torchvision


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    def __init__(self, in_size=10, out_size=1, hidden_dim=32, norm_reduce=False):
        super(MLP, self).__init__()
        self.norm_reduce = norm_reduce
        self.model = nn.Sequential(
                            nn.Linear(in_size, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, out_size),
                            )
    def forward(self, x):
        out = self.model(x)
        if self.norm_reduce:
            out = torch.norm(out)

        return out

class ContextNet(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (kernel_size - 1) // 2

        self.context_net = nn.Sequential(
                                nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=padding),
                                nn.BatchNorm2d(hidden_dim),
                                nn.ReLU(),
                                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                                nn.BatchNorm2d(hidden_dim),
                                nn.ReLU(),
                                nn.Conv2d(hidden_dim, out_channels, kernel_size, padding=padding)
                            )


    def forward(self, x):
        out = self.context_net(x)
        return out

class ConvNet(nn.Module):
    def __init__(self, num_classes=10, num_channels=3, smaller_model=True, hidden_dim=128, return_features=False, **kwargs):
        super(ConvNet, self).__init__()

        kernel_size = 5

        self.smaller_model = smaller_model
        padding = (kernel_size - 1) // 2
        if smaller_model:
            print("using smaller model")
            self.conv1 = nn.Sequential(
                            nn.Conv2d(num_channels, hidden_dim, kernel_size),
                            nn.BatchNorm2d(hidden_dim),
                            nn.ReLU(),
                            nn.MaxPool2d(2)
                        )
        else:
            print("using larger model")
            self.conv0 = nn.Sequential(
                        nn.Conv2d(num_channels, hidden_dim, kernel_size, padding=padding),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU(),
                    )

            self.conv1 = nn.Sequential(
                            nn.Conv2d(hidden_dim, hidden_dim, kernel_size),
                            nn.BatchNorm2d(hidden_dim),
                            nn.ReLU(),
                            nn.MaxPool2d(2)
                        )


        self.conv2 = nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        self.final = nn.Sequential(
                    nn.Linear(hidden_dim, 200),
                    nn.ReLU(),
                    Identity() if return_features else nn.Linear(200, num_classes)
                  )
        self.num_features = 200


    def forward(self, x):
        """Returns logit with shape (batch_size, num_classes)"""

        # x shape: batch_size, num_channels, w, h

        if self.smaller_model:
            out = self.conv1(x)
        else:
            out = self.conv0(x)
            out = self.conv1(out)
        out = self.conv2(out)
        out = self.adaptive_pool(out) # shape: batch_size, hidden_dim, 1, 1
        out = out.squeeze(dim=-1).squeeze(dim=-1) # make sure not to squeeze the first dimension when batch size is 0.
        out = self.final(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_channels, num_classes, model_name, pretrained=None,
                 avgpool=False, return_features=False):
        super(ResNet, self).__init__()

        self.model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        self.num_features = self.model.fc.in_features
        if return_features:
            self.model.fc = Identity()
        else:
            self.model.fc = nn.Linear(self.num_features, num_classes)

        # Change number of input channels from 3 to whatever is needed
        # to take in the context also.
        if num_channels != 3:
            model_inplanes = 64
            old_weights = self.model.conv1.weight.data
            self.model.conv1 = nn.Conv2d(num_channels, model_inplanes,
                             kernel_size=7, stride=2, padding=3, bias=False)

            if pretrained:
                for i in range(num_channels):
                    self.model.conv1.weight.data[:, i, :, :] = old_weights[:, i % 3, :, :]

        if avgpool:
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        out = self.model(x)
        return out



