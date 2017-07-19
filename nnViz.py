# todo color with pos & negative
# table with samples
# color width 
# + plot bias
from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np

vertical_distance_between_layers = 4
horizontal_distance_between_neurons = 2
neuron_radius = 0.5
number_of_neurons_in_widest_layer = 4 


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y      

    def draw(self):
        global neuron_radius
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=True)
        circle.set_edgecolor((0,0,0))
        circle.set_facecolor((1,1,1))
        circle.set_linewidth(2)
        circle.set_zorder(2)
        pyplot.gca().add_patch(circle)



class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights
        self.maxWeight = 10


    def __intialise_neurons(self, number_of_neurons):
        global horizontal_distance_between_neurons
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        global horizontal_distance_between_neurons
        global number_of_neurons_in_widest_layer
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        global vertical_distance_between_layers
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        global neuron_radius
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)

        if linewidth > 0:
            lineColor = (0,0,.5)
        else:
            lineColor = (.5,0,0)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth, color=lineColor)
        line.set_zorder(1)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]

            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]

                    if self.previous_layer.weights is not None:
                        weight = np.min( (self.maxWeight, self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]) ) 
                    else:
                        weight = 1
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)
            neuron.draw()


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):        
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.show()


def visualize_model( model ):
    import tensorflow as tf
    nLayers = len(model.layers)

    modelWeightsTFVar = {}
    modelWeights = {}
    modelShape = {}
    for iLayer in range( nLayers ):
        modelShape[iLayer] = [ model.layers[iLayer].input_shape[1], model.layers[iLayer].output_shape[1] ]
        modelWeightsTFVar[iLayer] = model.layers[iLayer].weights

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iLayer in range( nLayers ):
            modelWeights[iLayer] = (modelWeightsTFVar[iLayer][0]).eval()
            #modelWeights[iLayer][1] = (modelWeightsTFVar[iLayer][1]).eval()    
    
    network = NeuralNetwork()
    
    # first layer
    network.add_layer( modelShape[0][0], np.transpose( modelWeights[0] ))

    for iLayer in range( nLayers - 1 ):    
        network.add_layer( modelShape[iLayer][1], np.transpose( modelWeights[iLayer+1]) )

    # last layer
    network.add_layer(modelShape[nLayers-1][1])

    network.draw()

if __name__ == "__main__":
    print('main')