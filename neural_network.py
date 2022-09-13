from PIL import Image
import numpy as np
from random import randint

class Layer:
    def __init__(self, num_nodes, num_outputs) -> None:
        self.num_nodes = num_nodes
        self.num_outputs = num_outputs

        self.biases = [0 for _ in range(self.num_outputs)]
        self.weights = [[0 for _ in range(self.num_outputs)] for _ in range(self.num_nodes)]


    def calculate_outputs(self, input):
        weighted_inputs = []

        for idx1 in range(self.num_outputs):
            output = self.biases[idx1]

            for idx2 in range(self.num_nodes -1):
                output += input[idx2] * self.weights[idx2][idx1]

            weighted_inputs.append(output)

        return weighted_inputs
                

class NeuralNetwork:
    def __init__(self) -> None:
        self.input_size = 2
        self.output_size = 2

        self.hidden_layers_amount = 1
        self.hidden_layers_size = 3

        # TODO Right now only works with 1 hidden layer
        self.layers = [Layer(self.hidden_layers_size, self.output_size) for _ in range(self.hidden_layers_amount)]
        self.layers.append(Layer(self.output_size, self.output_size))

        self.visualise()


    def classify(self, input):
        output = self.calculate_outputs(input)
        if output.index(max(output)) == 0: return (0, 0, 255)
        else: return (255, 0, 0)

        #return output.index(max(output))

    def calculate_outputs(self, input):
        for layer in self.layers:
            input = layer.calculate_outputs(input)

        return input


    def visualise(self):
        image_array = []

        for idx in range(750):
            row = []
            for idx2 in range(1000):
                row.append(self.classify([idx, idx2]))
            image_array.append(row)

        image = Image.fromarray(np.array(image_array).astype(np.uint8))
        image.show()

if __name__ == "__main__":
    NeuralNetwork()