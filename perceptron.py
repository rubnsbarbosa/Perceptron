from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np

class Perceptron():
	def __init__(self, num_iterations = 1000, learning_rate = 0.01):
		self.epochs = num_iterations
		self.eta = learning_rate
		self.weight = None
		self.bias = None

	def fit(self, X, y):
		num_rows, num_features = X.shape
		self.weight = np.zeros(num_features)
		self.bias = 0
		# make sure that y = 1 or y = 0
		y = self.normalize(y)
	
		for _ in range(self.epochs):
			for i, x in enumerate(X):
				u = np.dot(x, self.weight) + self.bias
				y_hat = self.activation_function(u)
					
				self.weight = self.weight + self.eta * (y[i] - y_hat) * x
				self.bias = self.bias + self.eta * (y[i] - y_hat)
				

	def predict(self, X):
		u = np.dot(X, self.weight) + self.bias
		y_hat = self.activation_function(u)
		return y_hat

	def activation_function(self, x):
                return np.where(x >= 0, 1, 0)

	def normalize(self, y):
		for i in range(len(y)):
			if y[i] > 0:
				y[i] = 1
			else:
				y[i] = 0
		return y

	def accuracy_score(self, y, hypothesis):
		number_success = 0
		for i in range(len(y)):
			if hypothesis[i] == y[i]:
				number_success += 1
		success_percentage = (number_success / len(y)) * 100
		return success_percentage


if __name__ == "__main__":
	X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=40)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

	p = Perceptron()
	p.fit(X_train, y_train)
	hypothesis = p.predict(X_test)
	accuracy = p.accuracy_score(y_test, hypothesis) 

	print('Accuracy: {}'.format(accuracy))
