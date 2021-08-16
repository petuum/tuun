from torchvision.models import resnet34, resnet50


def classificationResnet34(pretrained=False, **kwargs):
    model = resnet34(pretrained=pretrained, **kwargs)
    return model


def classificationResnet50(pretrained=False, **kwargs):
    model = resnet50(pretrained=pretrained, **kwargs)
    return model
