# Import libraries
from math import exp, pow
from random import randint
import matplotlib.pyplot as plt

# Import files
from artificial_task import Dataset

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

    # Visualizes all datapoints and the network's guess as a colourmap
    def visualise(self):
        # Set graph limitation
        plt.xlim([0, 100])
        plt.ylim([0, 100])


        # Visualizing the dataset
        for data in self.training_data:
            # Colour of the scatter
            if data[0] == 0: colour = self.safe_colour
            else: colour = self.danger_colour

            # Place the scatter
            plt.scatter((data[1]*100,), (data[2]*100,), s=(7,), c=colour)


        # Visualizing normalised data as a colour map
        colour_map = []
        for i in range(101):
            normalI = i
            colour_row = []
            for j in range(101):
                normalJ = j
                colour_row.append(self.classify([normalI, normalJ]))
            colour_map.append(colour_row)


        # Add the colour map and show the plot
        plt.imshow(colour_map)
        plt.show()



if __name__ == "__main__":
    NeuralNetwork()