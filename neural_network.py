# Import libraries
import numpy as np
from math import exp, pow
import matplotlib.pyplot as plt

# Import files
from dataset import Dataset

class Layer:
    def __init__(self, num_nodes_in, num_nodes_out) -> None:
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out

        self.biases = [0 for _ in range(self.num_nodes_out)]
        self.weights = [[0 for _ in range(self.num_nodes_out)] for _ in range(self.num_nodes_in)]

    def calculate_outputs(self, input):
        activations = []

        for node_out in range(self.num_nodes_out):
            output = self.biases[node_out]

            for node_in in range(self.num_nodes_in):
                output += input[node_in] * self.weights[node_in][node_out]

            activations.append(self.activation_function(output))

        return activations

    # Sigmoid activation function
    def activation_function(self, weighted_input):
        return 1 / (1 + exp(-weighted_input))

    def node_cost(self, output_activation, expected_output):
        error = output_activation - expected_output
        return error * error


class NeuralNetwork:
    def __init__(self) -> None:
        self.learn_rate = 0.3

        self.input_size = 2
        self.output_size = 2

        self.hidden_layers_amount = 0
        self.hidden_layers_size = 3

        self.generate_network()

        self.training_data = Dataset().dataset
        self.safe_colour = ("#0000ff",)
        self.danger_colour = ("#ff0000",)

        print(self.cost(self.training_data))
        self.visualise()

    # Data point = [int, float, float]
    def cost(self, data_point):
        output = self.calculate_outputs([data_point[1], data_point[2]])
        output_layer = self.layers[-1]

        cost = 0
        for node_out in range(len(output)):
            # ? Maybe a mistake here
            cost += output_layer.node_cost(output[node_out], data_point[0])

        return cost

    # Generates the correct amount of hidden layers/output that are defined in init
    def generate_network(self):
        self.layers = []
        for i in range(self.hidden_layers_amount):
            if i == 0:
                self.layers.append(Layer(self.input_size, self.hidden_layers_size))
            elif i == self.hidden_layers_amount:
                self.layers.append(Layer(self.hidden_layers_size, self.output_size))
            else:
                self.layers.append(Layer(self.hidden_layers_size, self.hidden_layers_size))

        if self.hidden_layers_amount > 0:
            self.layers.append(Layer(self.hidden_layers_size, self.output_size))
        else:
            self.layers.append(Layer(self.input_size, self.output_size))
            
    def classify(self, input):
        output = self.calculate_outputs(input)
        if output[0] > output[1]: return (122, 133, 255)
        else: return (255, 122, 122)

    def calculate_outputs(self, input):
        for layer in self.layers:
            input = layer.calculate_outputs(input)

        return input

    # Visualizes all datapoints and the network's prediction as a colourmap
    def visualise(self):
        image_size = 500
        plt.xlim([0, image_size])
        plt.ylim([0, image_size])

        # Visualizing normalised data as a colour map
        colour_map = np.ndarray(shape=(image_size, image_size, 3), dtype=np.uint8)
        for i in range(image_size):
            for j in range(image_size):
                colour_map[i, j] = self.classify([i/image_size, j/image_size])

        # Visualizing the dataset
        for data in self.training_data:
            # Colour of the scatter
            if data[0] == 0: colour = self.safe_colour
            else: colour = self.danger_colour

            # Place the scatter
            plt.scatter((data[1]*image_size,), (data[2]*image_size,), s=(10,), c=colour)
            
        # Add the colour map and show the plot
        plt.imshow(colour_map)
        plt.title(f"{image_size}x scaled (Higher Resolution)")
        plt.show()

    def cost(self, dataset):
        cost = 0
        for datapoint in dataset:
            output = self.calculate_outputs([datapoint[1], datapoint[2]])
            output_layer = self.layers[-1]

            for node in range(output_layer.num_nodes_out):
                cost += output_layer.node_cost(output[node], datapoint[0])

        return cost / len(dataset)


if __name__ == "__main__":
    NeuralNetwork()