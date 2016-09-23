import numpy as np
import random 


def randomize_input(X,y):
	combined = np.c_[ X, y] 
	print "combined shape"
	print combined.shape
	random.shuffle(combined)

	return combined[: , [0, combined.shape[1]-2] ], combined[: , combined.shape[1] -1 ]

class CrossValidation:

	def __init__(self, numSets, X, y): 
		self.k = numSets
		self.data, self.output = X, y


	def get_partitioned_data(self):

		print "before and after"
		print self.data.shape
		print self.output.shape

		#self.data, self.output = randomize_input(self.data, self.output)

		print self.data.shape 

		print self.output.shape 
		x_partitioned = np.array_split(self.data, self.k)
		y_partitioned = np.array_split(self.output, self.k)
		print x_partitioned[0].shape
		print y_partitioned[0].shape

		return x_partitioned, y_partitioned



