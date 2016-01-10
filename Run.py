import cv2
import json
import gzip
import cPickle
import Network

#Dataset URL: http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
#train_set, valid_set, test_set format: list(list(array(input_data)), list(target_numbers))
#input is an numpy.ndarray of 2 dimensions who's row's correspond to an example.
#target is a numpy.ndarray of 1 dimension that have the same length as the number of rows in the input. It should give the target to the example with the same index in the input.

dataset = 'mnist.pkl.gz'
with gzip.open(dataset, 'rb') as f:
	train_set, valid_set, test_set = cPickle.load(f)
	#Transform into list(tuple(input_data, target_number))
	train = [(x,y) for x,y in zip(train_set[0], train_set[1])]
	valid = [(x,y) for x,y in zip(valid_set[0], valid_set[1])]
	test  = [(x,y) for x,y in zip(test_set[0], test_set[1])]

configs_file = 'hyperparameters.json'
configs = json.load(open(configs_file, 'r'))

network_shape   = configs['network_shape']
learning_rate   = configs['learning_rate']
num_epochs	    = configs['num_epochs']
mini_batch_size = configs['mini_batch_size']

#Debug function for checking visual of data
def renderImg(imgSet, i):
	img = imgSet[0][i].reshape(28,28)
	cv2.imshow('Number', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	print '> Initialising Neural Network...'
	print 'Input  Neurons:', network_shape[0]
	print 'Hidden Neurons:', network_shape[1]
	print 'Output Neurons:', network_shape[2]
	neural_net = Network.Network(network_shape)
	print '> Neural Network created!'
	print '> Training Network...'
	print 'Learning Rate:', learning_rate
	print 'Number of Epochs:', num_epochs
	neural_net.StochasticGradientDescent(train, num_epochs, mini_batch_size, learning_rate, test_data=valid)
	print '> Training Finished!'