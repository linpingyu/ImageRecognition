import preprocess
import prediction
import argparse
import torch



def Main():
	parser = argparse.ArgumentParser(description='This is a image identifier prediction program.')
	parser.add_argument('image', 
                    help='Please specify your path to the single image you want to predict on', type=str)
	parser.add_argument('checkpoint',
                   help='Please specify the directory of your checkpoint file',
                   type=str)

	parser.add_argument('--top_k',
                   help='The number of top k predicted categories of the image',
                   type=int,
                   default='5')
	parser.add_argument('--gpu',help='enter GPU mode', action='store_true')
	parser.add_argument('--category_names',help='Path to mapping json file', action='store', default = '')
	args = parser.parse_args()


	# GPU setup
	if args.gpu:
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	else:
		device = 'cpu'

	print('Loading model...\n')

	model = prediction.load_model(args.checkpoint, device)

	print('Predicting...\n')

	probs, classes = prediction.predict(args.image, model, device, args.top_k)

	if len(args.category_names) > 0:
		classes = prediction.mapping(classes, args.category_names)

	for i in range(len(classes)):
		print("The probability of being {} is {:.2f}%.\n".format(classes[i], probs[i] * 100))


if __name__ == '__main__':
	Main()