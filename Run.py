import cv2
import json
import DataLoader
import Network
import numpy as np
import matplotlib.pyplot as plt

configs = None
train_data = None
valid_data = None
test_data  = None

def init():
	global configs, train_data, valid_data, test_data
	configs_file = 'hyperparameters.json'
	configs = json.load(open(configs_file, 'r'))
	#Dataset URL: http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
	dataset = 'mnist.pkl.gz'
	data_loader = DataLoader.DataLoader(dataset)
	train_data, valid_data, test_data = data_loader.LoadData()

#Debug function for checking visual of data
def RenderImg(imgSet, i):
	img = imgSet[i][0].reshape(28,28)
	cv2.imshow('Number', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def PlotEpochAccuracy(epochs, hist):
	fig = plt.figure(facecolor='white')
	fig.canvas.set_window_title('Network accuracy per epoch')
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(range(epochs), hist, color='#1F77B4')
	ax.set_xlabel(r'$epochs$')
	ax.set_ylabel(r'$accuracy$')
	plt.show()

if __name__ == '__main__':
	init()
	#RenderImg(train_data, 1)
	print '> Initialising Neural Network...'
	print 'Input  Neurons:', configs['network_shape'][0]
	print 'Hidden Neurons:', configs['network_shape'][1]
	print 'Output Neurons:', configs['network_shape'][2]
	neural_net = Network.Network(configs['network_shape'])
	print '> Neural Network created!'
	print '> Training Network...'
	print 'Learning Rate:', configs['learning_rate']
	print 'Number of Epochs:', configs['num_epochs']
	epoch_accuracy = neural_net.StochasticGradientDescent(
		train_data,
		configs['num_epochs'],
		configs['mini_batch_size'],
		configs['learning_rate'],
		test_data=test_data
	)
	print '> Training Finished!'
	print '> Running Test Data on Network...'
	accuracy = (float(neural_net.Evaluate(test_data))/float(len(test_data)))*100
	print '> Finished Classifying {0} Images'.format(len(test_data))
	print '> Network Accuracy: {0:.2f}%'.format(accuracy)
	PlotEpochAccuracy(configs['num_epochs'], epoch_accuracy)