# todo color with pos & negative
# table with samples
# color width 
# + plot bias
from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np

verticalDistanceBetweenLayers = 5.5
horizontalDistanceBetweenNeurons = 2
neuronRadius = 0.75
nNeuronsInWidestLayer = 4 

class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y      

    def draw(self):
        global neuronRadius
        circle = pyplot.Circle((self.x, self.y), radius=neuronRadius, fill=True)
        circle.set_edgecolor((0,0,0))
        circle.set_facecolor((1,1,1))
        circle.set_linewidth(2)
        circle.set_zorder(2)
        pyplot.gca().add_patch(circle)

class Layer():
    def __init__(self, network, nNeurons, weights):
        self.prevLayer = self.get_prevLayer(network)
        self.y = self.compute_layer_vertical_pos()
        self.neurons = self.initialize_neurons(nNeurons)
        self.weights = weights
        self.maxWeight = 5
        self.minWeight = .1

    def initialize_neurons(self, nNeurons):
        global horizontalDistanceBetweenNeurons
        neurons = []
        x = self.compute_left_margin(nNeurons)
        for iteration in range(nNeurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontalDistanceBetweenNeurons
        return neurons

    def compute_left_margin(self, nNeurons):
        global horizontalDistanceBetweenNeurons
        global nNeuronsInWidestLayer
        return horizontalDistanceBetweenNeurons * (nNeuronsInWidestLayer - nNeurons) / 2

    def compute_layer_vertical_pos(self):
        global verticalDistanceBetweenLayers
        if self.prevLayer:
            return self.prevLayer.y + verticalDistanceBetweenLayers
        else:
            return 0

    def get_prevLayer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def draw_edge(self, neuron1, neuron2, linewidth, sign):
        global neuronRadius
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        xOffset = neuronRadius * sin(angle)
        yOffset = neuronRadius * cos(angle)
        lineXTuple = (neuron1.x - xOffset, neuron2.x + xOffset)
        lineYTuple = (neuron1.y - yOffset, neuron2.y + yOffset)

        if sign > 0:
            lineColor = (0, 0, .5, .5) # positive weights in red, 50% transparency
        else:
            lineColor = (.5, 0, 0, .5) # negative weights in blue, 50% transparency

        line = pyplot.Line2D(lineXTuple, lineYTuple, linewidth=linewidth, color=lineColor)
        line.set_zorder(1)
        pyplot.gca().add_line(line)

    def draw(self):
        for iNeuronCurrentLayer in range(len(self.neurons)):
            neuron = self.neurons[iNeuronCurrentLayer]

            if self.prevLayer:
                for iNeuronPreviousLayer in range(len(self.prevLayer.neurons)):
                    prevLayer_neuron = self.prevLayer.neurons[iNeuronPreviousLayer]

                    if self.prevLayer.weights is not None:
                        rawWeight = self.prevLayer.weights[iNeuronCurrentLayer, iNeuronPreviousLayer]
                        if rawWeight > 0:
                            sign = 1
                        else:
                            sign = -1

                        processedWeight = abs( self.prevLayer.weights[iNeuronCurrentLayer, iNeuronPreviousLayer])
                        
                        weight = min ( (self.maxWeight , max( self.minWeight,  processedWeight)) )
                        
                        #print ( str(weight) + ', ' + str( rawWeight ))
                    else:
                        weight = self.minWeight
                    self.draw_edge(neuron, prevLayer_neuron, weight, sign)
            neuron.draw()



class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, nNeurons, weights=None):
        layer = Layer(self, nNeurons, weights)
        self.layers.append(layer)

    def draw(self):        
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.show()


def visualize_model( model ):    
    nLayers = len(model.layers)

    modelWeightsTFVar = {}
    modelWeights = {}
    modelShape = {}

    for iLayer in range( len( model.layers ) ):
        modelShape[iLayer] = [ model.layers[iLayer].input_shape[1], model.layers[iLayer].output_shape[1] ]
        modelWeights[iLayer] = model.layers[iLayer].get_weights()[0]
    
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