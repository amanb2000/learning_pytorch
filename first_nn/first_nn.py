import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.autograd import Variable

import csv

from random import shuffle


csvIn = csv.reader(open('iris.csv'), delimiter=",")

all_data = []

for row in csvIn:
	all_data.append(row)

class IrisNet(nn.Module): #All of the nets made in PyTorch are made as classes ("custom modules"), inheriting the nn.Module
	#You need two functions in each custom module: __init__ (specify input dimensions, hidden layer dimensions, number of
		#output classes) and forward (takes in self, the input vector), which is where you actually implement
			#each step in order :D
	def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
		super(IrisNet, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden1_size) #nn.Linear implements fully connected layer
			#takes in number of incoming connections and number of outgoing connections
			#has one bias per each outgoing edge
		self.relu1 = nn.ReLU()
			#making 1 relu activation function for each thing.
		self.fc2 = nn.Linear(hidden1_size, hidden2_size)
		self.relu2 = nn.ReLU()
		self.fc3 = nn.Linear(hidden2_size, num_classes)

	def forward(self, x): 
		out = self.fc1(x)
		out = self.relu1(out)
		out = self.fc2(out)
		out = self.relu2(out)
		out = self.fc3(out)
		return out

def accuracy(out, labels):
	outputs = np.argmax(out, axis=1)
	return np.sum(outputs==labels)/float(labels.size)



model = IrisNet(4, 100, 50, 3)
# print(model) #you can actually do this omg

all_data = all_data[1:len(all_data)]
# print(all_data[0:10])

shuffle(all_data)

# print("Length of all data:", len(all_data))

flower_name_dict = {"setosa":0, "versicolor":1, "virginica":2, }

for i in all_data:
	i[4] = flower_name_dict[i[4]]
	i[0] = float(i[0])
	i[1] = float(i[1])
	i[2] = float(i[2])
	i[3] = float(i[3])

# print("First 10 pieces of data: ")

# print(all_data[0:10])

training_set_list = all_data[0:100]
training_set = torch.Tensor(training_set_list)

# print("First few elements: ")
# print(training_set[0:10])

train_x = training_set[:, 0:4]
train_y = (training_set[:, 4]).long()

validation_set_list = all_data[101:len(all_data)]
validation_set = torch.Tensor(validation_set_list)

validate_x = validation_set[:, 0:4]
validate_y = (validation_set[:, 4]).long()


# loss function
loss_fn = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=0.9, dampening=0)
	#args: parameters to tune, learning rate, nsterov (TODO: research), momentum, dampening
	#Stochastic Gradient Descent

# Using the PyTorch Data loader


### TRAINING LOOPS! ###
#reminder: def'n of epoch: when the model has "seen" all the training data once.

num_epochs = 500

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

for epoch in range(num_epochs):
	model.train()
	optimizer.zero_grad()
	train_x_var = Variable(train_x)
	train_y_var = Variable(train_y)

	pred_y = model(train_x_var)

	loss = loss_fn(pred_y, train_y_var)
	loss.backward()
	optimizer.step()

	model.eval() #network in evaluation mode
	train_loss.append(float(loss.data))
	print((loss))


	train_correct = 0;
	train_total = 0;


print("NOW WE WILL TEST IT ON THE VALIDATION SET:")

plt.plot(train_loss)
plt.show()


total = 0
correct = 0

for i in range(len(validate_x)):
	print("The expected value is: ",validate_y[i])
	nn_output = model(validate_x[i])

	ind = torch.argmax(nn_output)

	print("The actual value was: ",ind);

	total += 1
	if ind == validate_y[i]:
		correct += 1
	print(" ")

print("The final accuracy on the validation set was: ", correct/total)

train_total = 0
train_correct = 0
for i in range(len(train_x)):
	nn_output = model(train_x[i])

	ind = torch.argmax(nn_output)

	train_total += 1
	if ind == train_y[i]:
		train_correct += 1



print("-=-==========")
print("The final accuracy was on the training set was: ", train_correct/train_total)


