import numpy as np
class CrossValidation:

	def __init__(self, numSets, X, y): 
		self.k = numSets
		self.data = X
		self.output = y

	def get_partitioned_data(self):
		x_partitioned = np.array_split(self.data, self.k)
		y_partitioned = np.array_split(self.output, self.k)
		print x_partitioned[0].shape
		print y_partitioned[0].shape

		return x_partitioned, y_partitioned