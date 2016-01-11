import gzip
import cPickle
import numpy as np

class DataLoader:

	def __init__(self, filename):
		with gzip.open(filename, 'rb') as f:
			self.train_set, self.valid_set, self.test_set = cPickle.load(f)

	def LoadData(self):
		train_input  = [np.reshape(x, (784, 1)) for x in self.train_set[0]]
		train_result = [self.VectorResult(y) for y in self.train_set[1]]
		train_data   = zip(train_input, train_result)

		valid_input  = [np.reshape(x, (784, 1)) for x in self.valid_set[0]]
		valid_data   = zip(valid_input, self.valid_set[1])

		test_input   = [np.reshape(x, (784, 1)) for x in self.test_set[0]]
		test_data	 = zip(test_input, self.test_set[1])
		return (train_data, valid_data, test_data)

	#Vectorise Int into (10, 1) vector representation, with 1.0 at jth positon and 0.0 elsewhere
	def VectorResult(self, j):
		vect = np.zeros((10, 1))
		vect[j] = 1.0
		return vect