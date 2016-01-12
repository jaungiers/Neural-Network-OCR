import random
import numpy as np

class Network:

	def __init__(self, structure):
		self.num_layers = len(structure)
		self.structure  = structure
		self.biases 	= [np.random.randn(y, 1) for y in structure[1:]]
		self.weights	= [np.random.randn(y, x) for x, y in zip(structure[:-1], structure[1:])]
		self.epoch_accuracy = []

	def FeedForward(self, a):
		for b, w in zip(self.biases, self.weights):
			a = self.Sigmoid(np.dot(w, a)+b)
		return a

	#Run SGC on the whole network
	#train_data is list of tuples in format (input, desired_output)
	#If test_data is provided, network will evaluate against it on each epoch. Slow!
	def StochasticGradientDescent(self, train_data, epochs, mini_batch_size, learn_rate, test_data=None):
		if test_data: n_test = len(test_data)
		n = len(train_data)
		for i in xrange(epochs):
			random.shuffle(train_data)
			mini_batches = [train_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]

			for batch in mini_batches:
				self.UpdateMiniBatch(batch, learn_rate)

			if test_data:
				accuracy = (float(self.Evaluate(test_data))/float(n_test))*100
				self.epoch_accuracy.append(accuracy)
				print 'e-{0}: {1:.2f}%'.format(i, accuracy)
			else:
				print 'e-{0} completed'.format(i)
		return self.epoch_accuracy

	#Apply Gradient Descent to mini batch
	#mini_batch is list of tuples in format (input, desired_output)
	#k is the hyperparameter learning rate
	def UpdateMiniBatch(self, mini_batch, k):
		batch_b = [np.zeros(b.shape) for b in self.biases]
		batch_w = [np.zeros(w.shape) for w in self.weights]

		for x, y in mini_batch:
			delta_batch_b, delta_batch_w = self.Backpropagation(x, y)
			batch_b = [b+delta_b for b, delta_b in zip(batch_b, delta_batch_b)]
			batch_w = [w+delta_w for w, delta_w in zip(batch_w, delta_batch_w)]

		self.weights = [w-(k/len(mini_batch))*bw for w, bw in zip(self.weights, batch_w)]
		self.biases  = [b-(k/len(mini_batch))*bb for b, bb in zip(self.biases, batch_b)]

	#Apply Backpropagation algorithm on whole network
	#x,y are the output given and output desired respectively
	#Returns a tuple (b, w) representing gradient of cost function C
	def Backpropagation(self, x, y):
		tau_b = [np.zeros(b.shape) for b in self.biases]
		tau_w = [np.zeros(w.shape) for w in self.weights]

		#FeedForward
		activation  = x
		activations = [x]
		vectors_z	= []

		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			vectors_z.append(z)
			activation = self.Sigmoid(z)
			activations.append(activation)

		#BackwardPass
		delta = self.CostDerivative(activations[-1], y) * self.Sigmoid(vectors_z[-1], True)
		tau_b[-1] = delta
		tau_w[-1] = np.dot(delta, activations[-2].transpose())

		for l in xrange(2, self.num_layers):
			z = vectors_z[-l]
			sigmoid_derivative = self.Sigmoid(z, True)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_derivative
			tau_b[-l] = delta
			tau_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (tau_b, tau_w)

	#Evaluate num of inputs which are correctly predicted.
	#Result output is the index of whichever neuron has the highest activation value
	def Evaluate(self, test_data):
		results = [(np.argmax(self.FeedForward(x)), y) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in results)

	def CostDerivative(self, output_activations, y):
		return (output_activations-y)

	def Sigmoid(self, z, deriv=False):
		if deriv:
			return self.Sigmoid(z)*(1-self.Sigmoid(z))
		return 1.0/(1.0+np.exp(-z))