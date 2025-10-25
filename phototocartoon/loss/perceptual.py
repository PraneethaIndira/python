import torch
import torch.nn as nn
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    def __init__(self, selected_layers=None):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:36] # Up to relu4_2
        # 🔧 Replace in-place ReLU to avoid autograd errors

        for i, layer in enumerate(vgg):
            if isinstance(layer, nn.ReLU):
                vgg[i] = nn.ReLU(inplace=False)  # ✅ safe for autograd
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.selected_layers = selected_layers or ['relu3_3', 'relu4_2']
        self.layer_map = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15,
            'relu4_1': 20, 'relu4_2': 22
        }

    def forward(self, x):
        features = {}
        for name, layer_idx in self.layer_map.items():
            x = self.vgg[layer_idx](x)
            if name in self.selected_layers:
                features[name] = x
        return features

class PerceptualLoss(nn.Module):
    def __init__(self, weight=1.0, selected_layers=None):
        super(PerceptualLoss, self).__init__()
        self.weight = weight
        self.extractor = VGGFeatureExtractor(selected_layers)
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        input_features = self.extractor(input)
        target_features = self.extractor(target)
        loss = 0.0
        for layer in self.extractor.selected_layers:
            loss += self.criterion(input_features[layer], target_features[layer])
        return self.weight * loss