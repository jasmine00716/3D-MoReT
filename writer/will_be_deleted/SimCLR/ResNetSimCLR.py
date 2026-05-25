import torch.nn as nn
import torchvision.models as models

#
# class ResNetSimCLR(nn.Module):
#
#     def __init__(self, base_model, out_dim):
#         super(ResNetSimCLR, self).__init__()
#         self.base_model = base_model
#         self.simclr_model = SimCLR(base_model.fc.in_features, out_dim)
#
#
#     def forward(self, x):
#         # feature extraction
#         self.base_model(x)

class SimCLRWithSegmentation(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(SimCLRWithSegmentation, self).__init__()

        # SimCLR backbone (ResNet50 in this case)
        self.simclr_backbone = SimCLR(base_model=models.resnet50(pretrained=True), projection_dim=128)

        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),  # Adjust input channels based on ResNet50's final output
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Getting features from the SimCLR backbone
        features = self.simclr_backbone(x)

        # Segmentation head forward pass
        segmentation_output = self.segmentation_head(features)

        return segmentation_output





if __name__ == '__main__':
    # num_classes = 21
    # resnet = models.resnet50(pretrained=True)
    # resnet.fc = nn.Identity()
    #
    # resnetsimclr_model = ResNetSimCLR(resnet, num_classes)

    # Example usage
    num_classes = 21  # Number of segmentation classes
    model = SimCLRWithSegmentation(num_classes)
    print(model)