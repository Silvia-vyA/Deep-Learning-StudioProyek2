import timm
from torchvision import transforms

NUM_CLASSES = 5
IMG = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_model():
    # Arsitektur saja, tanpa mengunduh bobot
    return timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)


def get_transform():
    return transforms.Compose([
        transforms.Resize(292),
        transforms.CenterCrop(IMG),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
