import numpy as np

class NeuralNetwork:
    def __init__(self, I_num, H1_num, H2_num, O_num):
        # initialize node num
        self.I_num = I_num
        self.H1_num = H1_num
        self.H2_num = H2_num
        self.O_num = O_num
        # initialize weights
        # self.W_ih=np.random.normal(0.0, pow(H_num,-0.5), (I_num,H_num))
        # self.W_ho=np.random.normal(0.0, pow(O_num,-0.5), (H_num,O_num))
        self.W1 = np.random.rand(I_num, H1_num) - 0.5
        self.W2 = np.random.rand(H1_num, H2_num) - 0.5
        self.W3 = np.random.rand(H2_num, O_num) - 0.5

    def S(self, x):
        return 1.0 / (1 + np.e ** (-x))

    def train(self, inputs, targets, alpha=0.1):
        inputs = np.array(inputs,ndmin=2)
        targets = np.array(targets,ndmin=2)

        H1_i = inputs @ self.W1
        H1_o = self.S(H1_i)
        H2_i = H1_o @ self.W2
        H2_o = self.S(H2_i)
        O_i = H2_o @ self.W3
        O_o = self.S(O_i)

        E3 = targets - O_o
        E2 = E3 @ self.W3.T
        E1 = E2 @ self.W2.T
        self.W3 += alpha * (H2_o.T @ (E3 * O_o * (1 - O_o)))
        self.W2 += alpha * (H1_o.T @ (E2 * H2_o * (1 - H2_o)))
        self.W1 += alpha * (inputs.T @ (E1 * H1_o * (1 - H1_o)))

    def query(self, inputs):
        inputs = np.array(inputs, ndmin=2)

        H1_i = inputs @ self.W1
        H1_o = self.S(H1_i)
        H2_i = H1_o @ self.W2
        H2_o = self.S(H2_i)
        O_i = H2_o @ self.W3
        O_o = self.S(O_i)
        return O_o


I_num = 784
H1_num = 100
H2_num = 100
O_num = 10

neuralNetwork = NeuralNetwork(I_num, H1_num, H2_num, O_num)

data_file = open('mnist_test.csv')
data_list = data_file.readlines()
data_file.close()

n = len(data_list)
train_size = int(n * 0.9)
test_size = n - train_size
train_list = data_list[:train_size]
test_list = data_list[-test_size:]


epoch = 1
for i in range(epoch):
    for record in train_list:
        all_values = record.split(',')
        inputs = np.asfarray(all_values[1:]) / 255.0
        targets = np.zeros(O_num)
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