import model_train
import preprocess
import argparse
from torchvision import datasets
import torch


def Main():
	parser = argparse.ArgumentParser(description='This is a image identifier training program.')
	parser.add_argument('data_dir', 
                    help='Please specify your directory of images you want to use for training the model.'+ \
                    'Please notice that there must be three sub-directories within the directory given,'+ \
                    'named train, test, validation respectively', type=str)
	parser.add_argument('--save_dir',
                   help='Please specify the directory that you want to store you model checkpoint file',
                   type=str,
                   action='store',
                   dest = 'save_directory',
                   default = 'checkpoint.pth')

	parser.add_argument('--arch',
                   help='Please specify the architecture you want to use',
                   type=str,
                   action='store',
                   default='vgg16')
	parser.add_argument('--gpu',help='enter GPU mode', action='store_true')
	parser.add_argument('--hidden_units', help='number of hidden_units', type=int, default=8192)
	parser.add_argument('--epochs', help='number of epochs', type=int, default=5)
	parser.add_argument('--learning_rate', help='learning rate', default=0.0001, type=float)


	args = parser.parse_args()


	# Directory setup
	data_dir = args.data_dir
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'


	# GPU setup
	if args.gpu:
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	else:
		device = 'cpu'
        
	# load train/validation/test datasets and loaders

	print('Loading images...\n')
    
	train_datasets = datasets.ImageFolder(train_dir, preprocess.train_transforms())        
	valid_datasets = datasets.ImageFolder(valid_dir, preprocess.test_validation_transforms())
	test_datasets = datasets.ImageFolder(test_dir, preprocess.test_validation_transforms())

	trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
	testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)
	validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
    
	print('Initializing Architecture...\n')
    
	model = model_train.get_pretrained_model(arch = args.arch)
	for params in model.parameters():
		params.requires_grad = False


	classifier = model_train.def_classifier(args.hidden_units)
	model.classifier = classifier
	model.to(device)

	criterion = model_train.criterion()
	optimizer = model_train.optimizer(learning_rate = args.learning_rate, params = model.classifier.parameters())
	steps = 0
	print_every = 40
    
	print('Training...\n')
    
	model_train.train_model(model, criterion, optimizer, trainloader, validloader, device, steps, print_every, args.epochs)
    
	model.to('cpu')

	model_train.save_checkpoint(model, optimizer, train_datasets, arch = args.arch, save_directory = args.save_directory)


if __name__ == '__main__':
	Main()