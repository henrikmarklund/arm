
import torch
import torch.nn as nn
import torchvision

class ContextNet(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, classify=False):
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

        # The following three lines are not used for any of the experiments in the paper.
        self.classify = classify
        if classify:
            self.classifier = nn.Linear(out_channels, 4)

    def forward(self, x):

        out = self.context_net(x)


        # The following three lines are not used for any experiments in the paper.
        if self.classify:
            out = torch.mean(out, dim=(-1,-2))
            out = self.classifier(out)

        return out

class ResNet(nn.Module):

    def __init__(self, num_channels, num_classes, model_name, pretrained=None,
                 avgpool=False):
        super(ResNet, self).__init__()

        self.model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

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


class ContextualConvNet(nn.Module):

    def __init__(self, num_channels=3, prediction_net='resnet18', num_classes=10,
                 support_size=50, use_context=None, n_context_channels=None,
                 pretrained=None, context_net='convnet',  **kwargs):
        super(ContextualConvNet, self).__init__()

        self.num_channels = num_channels
        self.support_size = support_size
        self.use_context = use_context

        # This is by default just 3.
        if n_context_channels is None:
            self.n_context_channels = num_channels
        else:
            self.n_context_channels = n_context_channels

        if use_context:
            self.context_net = ContextNet(num_channels, self.n_context_channels,
                                          hidden_dim=64, kernel_size=5)
            n_pred_channels = num_channels + n_context_channels
        else:
            n_pred_channels = num_channels

        if prediction_net == 'convnet':
            if use_context:
                cml = True
            else:
                cml = False
            self.prediction_net = ConvNet(num_channels=n_pred_channels, num_classes=num_classes, cml=cml)
        else: # resnets
            self.prediction_net = ResNet(num_channels=n_pred_channels,
                                          num_classes=num_classes, model_name=prediction_net,
                                         pretrained=pretrained)
        #self.prediction_net = nn.DataParallel(self.prediction_net)



    def forward(self, x):

        if self.use_context:

            batch_size, c, h, w = x.shape
            if batch_size % self.support_size == 0:
                meta_batch_size, support_size = batch_size // self.support_size, self.support_size
            else:
                meta_batch_size, support_size = 1, batch_size

            context = self.context_net(x) # Shape: batch_size, channels, H, W

            context = context.reshape((meta_batch_size, support_size, self.n_context_channels, h, w))
            context = context.mean(dim=1) # Shape: meta_batch_size, self.n_context_channels

            context = torch.repeat_interleave(context, repeats=support_size, dim=0) # meta_batch_size * support_size, context_size

            x = torch.cat([x, context], dim=1)

        out = self.prediction_net(x)

        return out

class ConvNet(nn.Module):
    def __init__(self, num_classes=10, num_channels=3, cml=True, **kwargs):
        super(ConvNet, self).__init__()

        hidden_dim = 128
        kernel_size = 5

        self.cml = cml
        padding = (kernel_size - 1) // 2
        if cml:
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
                            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding),
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
                    nn.Linear(200, num_classes)
                  )


    def forward(self, x):
        """Returns logit with shape (batch_size, num_classes)"""

        # x shape: batch_size, num_channels, w, h

        if self.cml:
            out = self.conv1(x)
        else:
            out = self.conv0(x)
            out = self.conv1(out)
        out = self.conv2(out)
        out = self.adaptive_pool(out).squeeze()
        out = self.final(out)

        return out

    def args_dict(self):
        """
            Model args used when saving and loading the model from a ckpt
        """

        #model_args = {
        #        'hidden_dim': self.hidden_dim,
        #        'num_classes': self.num_classes,
        #        'kernel_size': self.kernel_size,
        #        'num_channels': self.num_channels
        #        }
        model_args = {}

        return model_args

