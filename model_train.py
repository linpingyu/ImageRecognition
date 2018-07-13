import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict





# functions related to model training

def get_pretrained_model(arch = 'vgg16'):
    model = getattr(models, arch)(pretrained=True)
    return model
    

def def_classifier(hidden_units):
    layers = OrderedDict([
        ('fc1', nn.Linear(in_features=25088, out_features=hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.4)),
        ('fc2', nn.Linear(in_features=hidden_units, out_features=102)),
        ('output', nn.LogSoftmax(dim=1))
    ])
    
    classifier = nn.Sequential(layers)
    return classifier

def criterion():
    return nn.NLLLoss()

def optimizer(learning_rate, params):
    optimi = optim.Adam(lr = learning_rate, params = params)
    return optimi

def validation(model, loader, criterion, device):
    test_loss = 0
    accuracy = 0
    for img, lab in iter(loader):
        img, lab = img.to(device), lab.to(device)
        output = model.forward(img)
        test_loss += criterion(output, lab).item()
        
        ps = torch.exp(output)
        equality = (lab.data == ps.max(dim = 1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy

def train_model(model, criterion, optimizer, trainloader, validloader, device, steps, print_every, epochs = 5):
    for e in range(epochs):
        model.train()
        running_loss = 0
        for img, lab in iter(trainloader):
            steps += 1
            img, lab = img.to(device), lab.to(device)
            optimizer.zero_grad()
            output = model.forward(img)
            loss = criterion(output, lab)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0
                model.train()

def save_checkpoint(model, optimizer, train_datasets, arch, save_directory = 'checkpoint.pth'):
    checkpoint = {'epoch': 3,
                  'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'optimizer_state': optimizer.state_dict(),
                  'class_to_idx': train_datasets.class_to_idx,
                  'arch': arch}
    torch.save(checkpoint, save_directory)
    return 0

