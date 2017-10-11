import numpy as np
import matplotlib.pylab as plt
import warnings
warnings.filterwarnings("ignore")

from ipywidgets import interact, fixed, FloatSlider
import ipywidgets as widgets
from IPython.display import display

def sigmoid ( a ):
    return 1./(1.+np.exp(-1. * a))

class universalNN ( object ):
    def __init__ ( self, rangeStart = 0 , rangeEnd = 1, nIntervals = 200 ):
        plt.figure()

        self.heights = []
        self.nPairs = 1
        
        self.rangeStart = rangeStart
        self.rangeEnd = rangeEnd
        self.nIntervals = nIntervals
        self.x = np.linspace( self.rangeStart, self.rangeEnd, self.nIntervals)
        
        self.targetFunctionText = 'np.sin(self.x*np.pi*3)'
        self.outputActivation = self.x
        
        self.weightLayers = {}
        self.biasWeights = {}

        self.updateTargetFunction()


    def updatePairs(self, nPairs):
        self.nPairs = int(nPairs)
        
        weightInputToHidden = 900       
        bias = 1

        nHidden = self.nPairs * 2
        biasSpan = np.max ( (20, int( self.nIntervals/(self.nPairs) * 2 )) )        
        biasOffset = np.min( (-20, -1.0 * round( self.nIntervals/(self.nPairs), 2 ) ) )
        
        if self.heights == []:            
            self.heights = np.ones((self.nPairs,1))*10
        if len(self.heights) != nPairs:            
            self.heights = np.ones((self.nPairs,1))*10

        weightLayers = {}
        biasWeights = {}
        weightLayers['inputToHidden'] = np.ones((nHidden, 1)) * weightInputToHidden
        weightLayers['hiddenToOutput'] = np.ones((nHidden, 1))
        biasWeights['inputToHidden'] = np.zeros((nHidden, 1))
        biasWeights['hiddenToOutput'] = np.ones((nHidden, 1)) * -(1.0/(self.nPairs)*3)

        hiddenActivations = {}
        outputActivation = 0

        for iPair in range (self.nPairs):
            iHidden = iPair*2
            biasWeights['inputToHidden'][iHidden] = iHidden * (-1. * biasSpan) + biasOffset
            biasWeights['inputToHidden'][iHidden+1] = (iHidden + 1) * (-1. * biasSpan) + biasOffset
            hiddenActivations[iHidden] = sigmoid( self.x*weightLayers['inputToHidden'][iHidden] + bias*biasWeights['inputToHidden'][iHidden] )
            hiddenActivations[iHidden+1] = sigmoid( self.x*weightLayers['inputToHidden'][iHidden+1] + bias*biasWeights['inputToHidden'][iHidden+1] )
            weightLayers['hiddenToOutput'][iHidden] = self.heights[iPair]
            weightLayers['hiddenToOutput'][iHidden+1] = -1.0 * self.heights[iPair]


        for iHidden in range (nHidden):
            outputActivation += hiddenActivations[iHidden] * weightLayers['hiddenToOutput'][iHidden] + bias * biasWeights['hiddenToOutput'][iHidden]

        self.weightLayers = weightLayers
        self.biasWeights = biasWeights

        self.outputActivation = sigmoid(outputActivation)
        self.update_plots( )
        
        
    def textUpdateHeights(self, sender):
        a = sender.value.split(',')
        for iHeight in range(len(a)):
            self.heights[iHeight] = float(a[iHeight].strip())
        self.updatePairs( self.nPairs )

    def textUpdateTargetFunction (self, sender):
        self.targetFunctionText = sender.value
        self.updateTargetFunction()
        
    def updateTargetFunction ( self ):        
        z = eval(self.targetFunctionText)
        z = z + np.abs(np.min(z))
        z = z/(np.max(z)+.0001)
        self.targetFunction = z
        self.update_plots()
                
    def update_plots ( self ): 
        plt.cla()
        plt.plot( self.targetFunction, 'r')
        plt.hold(True)
        plt.plot( self.outputActivation, 'b')

    def draw_network ( self ):
        import nnViz
        network = nnViz.NeuralNetwork()

        plt.figure( ) 
        plt.subplots_adjust( left=0.0, right=1.0, bottom=0.0, top=1.0 )
        network.add_layer(1, self.weightLayers['inputToHidden'] )
        network.add_layer(self.nPairs*2, np.transpose( self.weightLayers['hiddenToOutput'] ) ) 
        network.add_layer(1)
        network.draw()
        
        

if __name__ == "__main__":
    
    nn = universalNN()
    nn.updatePairs(20)
    
    nn.draw_network()
    
    