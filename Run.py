import cv2
import json
import cPickle
import Network

#Dataset URL: http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
#train_set, valid_set, test_set format: tuple(input, target)
#input is an numpy.ndarray of 2 dimensions who's row's correspond to an example.
#target is a numpy.ndarray of 1 dimension that have the same length as the number of rows in the input. It should give the target to the example with the same index in the input.

dataset = 'mnist.pkl'
f = open(dataset, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

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
	test = (train_set[0][0], train_set[1])
	neural_net.StochasticGradientDescent(train_set, num_epochs, mini_batch_size, learning_rate, test_data=valid_set)
	print '> Training Finished!'