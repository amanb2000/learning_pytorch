from mlxtend.data import loadlocal_mnist
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy
from tqdm import tqdm

from torch.autograd import Variable

X, y = loadlocal_mnist(
        images_path='train-images-idx3-ubyte', 
        labels_path='train-labels-idx1-ubyte')

print("The dim of X is:",X.shape)
print("The dim of y is:",y.shape)

ex_1 = X[0].view().reshape(28,28)

plt.imshow(ex_1)
plt.show()
print("The label for the shown image is: ",y[0])

X_test, y_test = loadlocal_mnist(
	images_path='t10k-images-idx3-ubyte',
	labels_path='t10k-labels-idx1-ubyte')

print("The style of y is: ")


class Simple_Net(nn.Module):
	def __init__(self, input_size, hidden_size, hidden_2_size, hidden_3_size, num_classes):
		super(Simple_Net, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, hidden_2_size)
		self.relu2 = nn.ReLU()
		self.fc3 = nn.Linear(hidden_2_size, hidden_3_size)
		self.relu3 = nn.ReLU()
		self.fc4 = nn.Linear(hidden_3_size, num_classes)

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu1(out)
		out = self.fc2(out)
		out = self.relu2(out)
		out = self.fc3(out)
		out = self.relu3(out)
		out = self.fc4(out)
		return out

def accuracy(out, labels):
	outputs = np.argmax(out, axis=1)
	return np.sum(outputs==labels)/float(labels.size)

def test_model(simple_rick, testor_y, testor_X, tensor_y, tensor_X):
	total = 0
	correct = 0

	for i in range(len(testor_y[0:1000])):
		nn_output = simple_rick(testor_X[i])
		ind = torch.argmax(nn_output)
		total += 1
		if ind == testor_y[i]:
			correct += 1

	# print("The final accuracy on validation set was: ",correct/total)
	validation_accuracy = correct/total
	# print("Now testing on Training Set: ")
	total = 0
	correct = 0

	for i in range(len(tensor_y[0:1000])):
		nn_output = simple_rick(tensor_X[i])
		ind = torch.argmax(nn_output)
		total += 1
		if ind == tensor_y[i]:
			correct += 1

	# print("The final accuracy on training set was: ",correct/total)
	train_accuracy = correct/total

	return [validation_accuracy, train_accuracy]

simple_rick = Simple_Net(784, 500, 300, 100, 10)
print(simple_rick)

tensor_X = torch.Tensor(X)
tensor_y = torch.Tensor(y).long()

testor_X = torch.Tensor(X_test)
testor_y = torch.Tensor(y_test).long()

Variable_X = Variable(tensor_X)
Variable_y = Variable(tensor_y)

# Variable_tX = Variable(testor_X)
# Variable_tY = Variable(testor_y)

loss_fn = nn.CrossEntropyLoss()

learning_rate = 0.001

optimizer = torch.optim.SGD(simple_rick.parameters(), lr=learning_rate, nesterov=True, momentum = 0.9, dampening = 0)

num_epochs =  5000

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

for epoch in range(num_epochs):
	simple_rick.train()
	optimizer.zero_grad()

	pred_y = simple_rick(Variable_X)
	loss = loss_fn(pred_y, Variable_y)
	loss.backward()
	optimizer.step()

	simple_rick.eval()
	train_loss.append(float(loss.data))


	test_results = test_model(simple_rick, testor_y, testor_X, tensor_y, tensor_X)

	train_accuracy.append(test_results[1])
	test_accuracy.append(test_results[0])

	if epoch % 100 == 0:
		print("=========="+str(epoch)+"==========")
		print("Loss: ",float(loss))
		print("Test Accuracy: ",test_results[0])
		print("Train Accuracy: ",test_results[1])

print("Displaying loss over time...")
plt.plot(train_loss, label='Training Loss')
plt.title("Training Loss Over Time")
plt.legend()
plt.show()
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(test_accuracy, label='Test Accuracy')
# plt.axis('equal')
plt.legend()
plt.axis
plt.show()










