import numpy as np
import scipy.special as ssp


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #количество узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #коэф обучения
        self.lrate = learningrate

        #weights
        self.wih = np.random.normal(0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        self.activ_func = lambda x: ssp.expit(x)

        pass

    def train(self, inputs_list, target_list):
        inputs = np.array(inputs_list, ndmin=2).T
        target = np.array(target_list, ndmin=2).T

        #сигнаалы скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activ_func(hidden_inputs)
        #сигналы выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activ_func(final_inputs)

        output_errors = target - final_outputs
        hidden_errors = np.dot(self.who, output_errors)

        self.who += self.lrate * np.dot(output_errors * final_outputs *
                                 (1-final_outputs), np.transpose(hidden_outputs))

        self.wih += self.lrate * np.dot(hidden_errors * hidden_outputs *
                                        (1-hidden_outputs), np.transpose(inputs))

    def query(self, inputs_list):
        #преобразовали в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activ_func(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activ_func(final_inputs)

        return final_outputs


if __name__ == '__main__':
    test_net = NeuralNetwork(3,3,3,0.3)

    x1 = [1, 2, 3]
    y1 = [2, 1, 1]

    test_net.train(x1, y1)

    print(test_net.query([0,1,3]))
