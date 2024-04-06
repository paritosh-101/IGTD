import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


# class ResNetSimCLR(nn.Module):

#     def __init__(self, base_model, out_dim, grayscale=False):
#         super(ResNetSimCLR, self).__init__()
#         self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
#                             "resnet34": models.resnet34(pretrained=False, num_classes=out_dim),  # Added ResNet34
#                             "resnet50": models.resnet50(pretrained=False, num_classes=out_dim),
#                             "resnet101": models.resnet101(pretrained=False, num_classes=out_dim),  # Added ResNet101
#                             "resnet152": models.resnet152(pretrained=False, num_classes=out_dim)  # Added ResNet152
#                             }

#         self.backbone = self._get_basemodel(base_model)

#         if grayscale:
#             # Modify the first convolutional layer to accept 1-channel images
#             self.backbone.conv1 = nn.Conv2d(1, self.backbone.conv1.out_channels,
#                                             kernel_size=self.backbone.conv1.kernel_size,
#                                             stride=self.backbone.conv1.stride,
#                                             padding=self.backbone.conv1.padding,
#                                             bias=False)


#         dim_mlp = self.backbone.fc.in_features

#         # dim_mlp = self.backbone.fc.in_features
#         # self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))

#         # add mlp projection head
#         self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

#     def _get_basemodel(self, model_name):
#         try:
#             model = self.resnet_dict[model_name]
#         except KeyError:
#             raise InvalidBackboneError(
#                 "Invalid backbone architecture. Choose one of: resnet18, resnet34, resnet50, resnet101, or resnet152.")
#         else:
#             return model

#     def forward(self, x):
#         return self.backbone(x)


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, grayscale=False):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet34": models.resnet34(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim),
                            "resnet101": models.resnet101(pretrained=False, num_classes=out_dim),
                            "resnet152": models.resnet152(pretrained=False, num_classes=out_dim)
                            }

        self.backbone = self._get_basemodel(base_model)

        # Replace 2D convolutions with 1D convolutions
        self.backbone = self._replace_conv2d_with_conv1d(self.backbone)

        # Modify the average pooling layer
        self.backbone.avgpool = nn.AdaptiveAvgPool1d(1)

        if grayscale:
            # Modify the first convolutional layer to accept single-channel images
            self.backbone.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Choose one of: resnet18, resnet34, resnet50, resnet101, or resnet152.")
        else:
            return model

    def _replace_conv2d_with_conv1d(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                conv1d = nn.Conv1d(child.in_channels, child.out_channels,
                                   kernel_size=child.kernel_size[0],
                                   stride=child.stride[0],
                                   padding=child.padding[0],
                                   bias=child.bias)
                setattr(module, name, conv1d)
            elif isinstance(child, nn.BatchNorm2d):
                bn1d = nn.BatchNorm1d(child.num_features)
                setattr(module, name, bn1d)
            else:
                self._replace_conv2d_with_conv1d(child)
        return module

    def forward(self, x):
        # # Permute dimensions from [B, C, H, W] to [B, H, W, C]
        # x = x.permute(0, 2, 3, 1)

        # # Reshape the tensor to [B, H, W]
        # x = x.reshape(x.size(0), x.size(1), x.size(2))

        # # Apply the modified ResNet backbone
        # x = self.backbone(x)
        
        # Reshape the tensor to [B, H, W]
        batch_size, height, width = x.size()[:3]
        x = x.view(batch_size, 1, height * width)

        # Apply the modified ResNet backbone
        x = self.backbone(x)
        
        return x