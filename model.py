import timm
import torch.nn as nn

def create_couple_model(num_classes=2, pretrained=True):
    # Using ResNet50 from timm
    model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)
    return model

if __name__ == "__main__":
    model = create_couple_model()
    print(model)
