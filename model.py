import torch.nn as nn
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    densenet121, DenseNet121_Weights
)


def get_model(model_name="mobilenetv2", num_classes=9):

    # ---------------- MobileNetV2 ----------------
    if model_name == "mobilenetv2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    # ---------------- ResNet50 ----------------
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    # ---------------- EfficientNet-B0 ----------------
    elif model_name == "efficientnetb0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    # ---------------- DenseNet121 ----------------
    elif model_name == "densenet121":
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    else:
        raise ValueError("Invalid model name. Choose from: mobilenetv2, resnet50, efficientnetb0, densenet121")

    return model





