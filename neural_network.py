# Import libraries
import matplotlib.pyplot as plt

# Import files
from artificial_task import Dataset

class Layer:
    def __init__(self, num_nodes, num_outputs) -> None:
        self.num_nodes = num_nodes
        self.num_outputs = num_outputs

        self.biases = [0 for _ in range(self.num_outputs)]
        self.weights = [[0 for _ in range(self.num_outputs)] for _ in range(self.num_nodes)]

    def calculate_outputs(self, input):
        weighted_output = []

        for node_output in range(self.num_outputs):
            output = self.biases[node_output]

            for node in range(self.num_nodes):
                output += input[node] * self.weights[node][node_output]

            weighted_output.append(output)

        return weighted_output
                

class NeuralNetwork:
    def __init__(self) -> None:
        self.input_size = 2
        self.output_size = 2

        self.hidden_layers_amount = 1
        self.hidden_layers_size = 3


        self.layers = [Layer(self.output_size, self.output_size)]
        #self.layers = [Layer(self.hidden_layers_size, self.output_size) for _ in range(self.hidden_layers_amount)]
        #self.layers.append(Layer(self.output_size, self.output_size))

        self.training_data = Dataset().dataset
        self.safe_colour = ("#0000ff",)
        self.danger_colour = ("#ff0000",)

        self.visualise()

    def classify(self, input):
        output = self.calculate_outputs(input)
        if output[0] > output[1]: return (122, 133, 255)
        else: return (255, 122, 122)

    def calculate_outputs(self, input):
        for layer in self.layers:
            input = layer.calculate_outputs(input)

        return input

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