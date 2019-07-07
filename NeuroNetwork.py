import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = self.sigmoid
        pass

    def sigmoid(self,x):
        return 1.0 / (1 + np.e ** (-x))

    def train(self, inputs_list, targets_list, alpha=0.1):
        inputs=np.array(inputs_list, ndmin=2).T
        targets=np.array(targets_list, ndmin=2).T

        # H_i=W_{i,h}*I
        hidden_inputs=np.dot(self.wih, inputs)
        # H_o=S(H_i)
        hidden_outputs=self.activation_function(hidden_inputs)

        # F_i=W_{h,o}*H_o
        final_inputs=np.dot(self.who, hidden_outputs)
        # F_o=S(F_i)
        final_outputs=self.activation_function(final_inputs)

        # Err_o=(T_o-F_o)
        output_errors=targets-final_outputs
        # Err_h=W_{h,o}^T*Err_o
        hidden_errors=np.dot(self.who.T,output_errors)

        # W_{h,o} += \alpha * Err_o * F_o(1-F_o) * H_o^T
        self.who += alpha * np.dot((output_errors * final_outputs * (1.0 -final_outputs)), np.transpose(hidden_outputs))
        # W_{i,h} += \alpha * Err_h * H_o(1-H_o) * I^T
        self.wih += alpha * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        # H_i=W_{i,h}*I
        hidden_inputs = np.dot(self.wih, inputs)
        # H_o=S(H_i)
        hidden_outputs = self.activation_function(hidden_inputs)

        # F_i=W_{h,o}*H_o
        final_inputs = np.dot(self.who, hidden_outputs)
        # F_o=S(F_i)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


input_nodes = 784
hidden_nodes = 100
output_nodes = 10

#learning_rate = 0.3



data_file = open('mnist_test.csv','r')
data_list = data_file.readlines()
data_file.close()
n = len(data_list)
training_num = int(n * 0.9)
test_num = n - training_num
training_data_list = data_list[:training_num]
# for record in training_data_list:
#     all_values = record.split(',')
#     inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
#     targets = np.zeros(output_nodes) + 0.01
#     targets[int(all_values[0])] = 0.99
#     neuralNetwork.train(inputs, targets)
#     pass

#test_data_file = open('mnist_test.csv','r')
test_data_list = data_list[-test_num:]#test_data_file.readlines()
#test_data_file.close()

alpha = 0.01
points = []
while alpha <= 1.0:
    neuralNetwork = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)

    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        neuralNetwork.train(inputs, targets, alpha)
        pass

    scorecard = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        outputs = neuralNetwork.query((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
        label = np.argmax(outputs)
        if label==correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
    score = sum(scorecard) * 1.0 / len(test_data_list)
    points.append([alpha,score])
    if alpha < 0.1:
        alpha = 0.1
    else:
        alpha += 0.1

points = np.array(points)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title('性能与学习率\n3层神经网络与MNIST数据集')
plt.xlabel('学习率')
plt.ylabel('性能')
plt.plot(points[:,0],points[:,1],'.-')
plt.show()


