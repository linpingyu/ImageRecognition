from torchvision import transforms
import numpy as np


# image pre-processing

def train_transforms():
    t = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return t

def test_validation_transforms():
    t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return t

def process_image(image):
    
    smaller = min(image.size)
    new_width = int(256 * image.width / smaller)
    new_height = int(256 * image.height / smaller)
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2
    image = image.resize((new_width, new_height)).crop((left, top, right, bottom))
    np_image = np.array(image)
    np_image = (np_image / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2,0,1))
    
    return np_image