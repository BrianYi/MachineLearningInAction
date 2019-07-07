import numpy as np


class NeuralNetwork:
    def __init__(self, I_num, H_num, O_num):
        # initialize node num
        self.I_num = I_num
        self.H_num = H_num
        self.O_num = O_num
        # initialize weights
        # self.W_ih=np.random.normal(0.0, pow(H_num,-0.5), (I_num,H_num))
        # self.W_ho=np.random.normal(0.0, pow(O_num,-0.5), (H_num,O_num))
        self.W_ih = np.random.rand(I_num, H_num) - 0.5
        self.W_ho = np.random.rand(H_num, O_num) - 0.5

    def S(self, x):
        return 1.0 / (1 + np.e ** (-x))

    def train(self, inputs, targets, alpha=0.1):
        inputs = np.array(inputs,ndmin=2)
        targets = np.array(targets,ndmin=2)

        H_i = inputs @ self.W_ih
        H_o = self.S(H_i)
        O_i = H_o @ self.W_ho
        O_o = self.S(O_i)

        E_o = targets - O_o
        E_h = E_o @ self.W_ho.T
        self.W_ho += alpha * (H_o.T @ (E_o * O_o * (1 - O_o)))
        self.W_ih += alpha * (inputs.T @ (E_h * H_o * (1 - H_o)))

    def query(self, inputs):
        inputs = np.array(inputs, ndmin=2)

        H_i = inputs @ self.W_ih
        H_o = self.S(H_i)
        O_i = H_o @ self.W_ho
        O_o = self.S(O_i)
        return O_o

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

neuralNetwork = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)

data_file = open('mnist_test.csv')
data_list = data_file.readlines()
data_file.close()

n = len(data_list)
train_size = int(n * 0.9)
test_size = n - train_size
train_list = data_list[:train_size]
test_list = data_list[-test_size:]


epoch = 5
for i in range(epoch):
    for record in train_list:
        all_values = record.split(',')
        inputs = np.asfarray(all_values[1:]) / 255.0
        targets = np.zeros(output_nodes)
        targets[int(all_values[0])] = 1
        neuralNetwork.train(inputs,targets)

correct_labels = []
predict_labels = []
for record in test_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    correct_labels.append(correct_label)
    inputs = np.asfarray(all_values[1:]) / 255.0
    outputs = neuralNetwork.query(inputs)
    label = np.argmax(outputs)
    predict_labels.append(label)
predict_labels = np.array(predict_labels)
correct_labels = np.array(correct_labels)
TrueCount = predict_labels==correct_labels
score = sum(TrueCount) * 1.0/len(TrueCount)
print(score)