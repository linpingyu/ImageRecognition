import numpy as np
import torch
from PIL import Image
import preprocess
import model_train


# functions related to model prediction

def load_model(filepath, device):

    checkpoint = torch.load(filepath)

    model = model_train.get_pretrained_model(checkpoint['arch'])
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model


def predict(image_path, model, device, topk=5):
    

    idx_map = {j:v for v,j in model.class_to_idx.items()}
    image = preprocess.process_image(Image.open(image_path))
    image = torch.from_numpy(image).unsqueeze(0).to(device).float()
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
    ps = torch.exp(output)
    
    probs, classes = ps.topk(topk)
    probs = probs.cpu().numpy()[0].tolist()
    classes = classes.cpu().numpy()[0].tolist()

    classes = [idx_map[i] for i in classes]
    
    
    return probs, classes

def mapping(classes, filepath = 'cat_to_name.json'):
    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        classes = [cat_to_name[i] for i in classes]
    return classes