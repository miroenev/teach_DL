''' X = time axis '''
''' Y = model width axis '''
''' Z = model height axis '''

from k3d import K3D as _3DBackend
import numpy as np
import matplotlib.pylab as _2DBackend

''' ------------
    DEFAULTS
------------ ''' 
globalTimestepLimit = 5

timeStepsToOmit = [] # range(2,8)


defaultInputStructure = { 'data': np.random.randint(0, 10, size=(100,1))/10.,
                            'dataIter': [],
                            'activeInputPositions': np.array([0, 0, 0], dtype='f') }
LSTM_flag = True
recurrentNeuronLimit = np.Inf

if LSTM_flag:
    recurrentLayers = [ 0, 1, 2 ]
    defaultArchStructure = { 'neuronsInLayers': [ 10, 10, 10, 1],
                                   'weights': [ np.random.randint(0, 10, size=(10, 10))/10. - .5,
                                                np.random.randint(0, 10, size=(10, 10))/10. - .5,
                                                np.random.randint(0, 10, size=(10, 1))/10. - .5 ],                                          
                                   'neuronPositions': {}}
else:
    '''
    # ARIMA like
    recurrentLayers = [] # [1]
    defaultArchStructure = { 'neuronsInLayers': [10, 1],
                            'weights': [ np.random.randint(0, 10, size=(10, 1))/10. - .5],
                            'neuronPositions': {}}
    '''
    '''
    # dense + no recurrence
    recurrentLayers = [] # [1]
    defaultArchStructure = { 'neuronsInLayers': [10, 10, 10, 1],
                                   'weights': [ np.random.randint(0, 10, size=(10, 10))/10. - .5,
                                                np.random.randint(0, 10, size=(10, 10))/10. - .5,
                                                np.random.randint(0, 10, size=(10, 1))/10. - .5 ],                                          
                            'neuronPositions': {}}
    
    '''

    
    recurrentLayers = [1] # [1]
    defaultArchStructure = { 'neuronsInLayers': [10, 5, 1],
                            'weights': [ np.random.randint(0, 10, size=(10, 5))/10. - .5,
                                        np.random.randint(0, 10, size=(5, 1))/10. - .5 ],
                            'neuronPositions': {}}
    


defaultTimeStructure = { 'timeIndex' : 0, 
                           'inputSamplesPerTimestep': 5 }


defaultPlottingParams = { '3D.Flag': True,
                             '3D.Shader': 'mesh', 
                             '2D.Backend': _2DBackend,
                             '3D.Backend': _3DBackend, 
                             'offset': np.array( [-20, 0, 6], dtype='f' ),
                         
                             'inputSampleSpacing': np.array( [0, .15, 0 ], dtype='f' ),
                             'timeSpacing': np.array( [8, 0, 0], dtype='f' ),
                         
                             'inputMarkerColor': 0x000000,
                             'inputActiveMarkerColor': 0x00FF00,
                             'inputLineColor': 0x999999,
                             'inputConnectionLineColor': 0xCCCCDD,
                             'positiveWeightColor': 0x9EFF96,
                             'negativeWeightColor': 0xCCAC91,
                             'recurrentWeightColor': 0x003366,
                         
                         
                             'neuronSize': 1,
                             'neuronColor': 0x9D6EFF,
                             'layerVerticalOffset': np.array([0, 0, 3], dtype='f'),
                             'interNeuronHorizontalSpacing': 1,
                             'minWeightLineThickness': .1,
                             'maxWeightLineThickness': 3,
                        }

''' --------
    CODE
---------''' 

class NNViz3D():
    
    def __init__ ( self,
                      inputStructure = defaultInputStructure,
                      archStructure = defaultArchStructure,
                      timeStructure = defaultTimeStructure,
                      plottingParams = defaultPlottingParams ):

        self.inputStructure = inputStructure
        self.archStructure = archStructure
        self.timeStructure = timeStructure
        self.plottingParams = plottingParams

        nTimesteps = int( len( inputStructure['data'] ) / timeStructure['inputSamplesPerTimestep'] )

        ''' initialize 2D or 3D display '''
        _3D_Plot = plottingParams['3D.Flag']
        
        if _3D_Plot:
            self.backend = plottingParams['3D.Backend']
            self.canvas = self.backend()
        else:
            self.backend = plottingParams['2D.Backend']                
            self.canvas = figure()
        
        self.draw_time_step ( )
        timeStructure['timeIndex'] += 1
        plottingParams['offset'] += plottingParams['timeSpacing']
        
        ''' plot each time step '''
        for iTimestep in range( 1, min( globalTimestepLimit, nTimesteps ) ):
            
            if iTimestep not in timeStepsToOmit:
                print ('-- timestep: ' + str( iTimestep ) + ' of ' + str ( nTimesteps) )
                self.draw_time_step ( )
                
                self.connect_to_prev_timestep ( )
                    
                plottingParams['offset'] += plottingParams['timeSpacing']
            else:
                plottingParams['offset'] += plottingParams['timeSpacing'] * .5
                        
            timeStructure['timeIndex'] += 1
            
        
        ''' update display '''
        if _3D_Plot:
            self.canvas.display()            
        else:
            self.canvas.show()
            
    def draw_time_step( self ):
        
        nInputDimensions = self.inputStructure['data'].shape[1]
        print(  '    input dimensions: ' + str ( nInputDimensions ) )

        ''' plot inputs '''
        for iInputDimension in range ( nInputDimensions ):            
            self.draw_input ( )
        
        ''' plot model '''
        self.draw_arch ( )
        
        ''' connect input to first layer '''
        self.connect_input_to_network ( )
        
        ''' connect up layers '''
        self.connect_layers_within_timestep ( ) 
        
        ''' connect up timesteps '''
        # one hot encoded rnn flags
    
    def draw_input ( self ):
        print(':: plotting input at timestep ')

        nSamples = len( self.inputStructure['data'] )
        self.inputStructure['activeInputPositions'] = np.array([0, 0, 0], dtype='f')

        
        # 1/2 Y offset -- where 1/2 is ( sampleSpacing * totalSamples )/ 2
        horizontalOffset = - self.plottingParams['inputSampleSpacing'] * int(nSamples/2.)   
        verticalOffset = - np.array( [ 0, 0, 5 ] ) # 2 Z units below current location
        
        offsetChain = self.plottingParams['offset'] + horizontalOffset + verticalOffset   
        
        activeRangeStart = self.timeStructure['timeIndex'] * self.timeStructure['inputSamplesPerTimestep']
        activeRangeEnd = activeRangeStart + self.timeStructure['inputSamplesPerTimestep']
        
        self.inputStructure['activeInputPositions'] = []

        for iSample in range ( nSamples-1 ):
            pointPosition = offsetChain + np.array( [0, 0, self.inputStructure['data'][iSample] ] )
            nextPointPosition = offsetChain + self.plottingParams['inputSampleSpacing'] \
                                    + np.array( [0, 0, self.inputStructure['data'][iSample + 1] ] )
            
            self.canvas += self.backend.line ( ( pointPosition[0], pointPosition[1], pointPosition[2],
                                          nextPointPosition[0], nextPointPosition[1], nextPointPosition[2] ), 
                                          color = self.plottingParams['inputLineColor'], width = .25 )

            if iSample >= activeRangeStart and iSample < activeRangeEnd:
                
                if self.inputStructure['activeInputPositions'] == []:
                    #self.inputStructure['activeInputPositions'] = pointPosition
                    self.inputStructure['activeInputPositions'] = np.expand_dims(pointPosition, axis = 0)
                    
                    
                else:
                    self.inputStructure['activeInputPositions'] = \
                        np.vstack( ( self.inputStructure['activeInputPositions'], 
                                     pointPosition ) )
                
                self.canvas += self.backend.points( positions = pointPosition, 
                                             color = self.plottingParams['inputActiveMarkerColor'], 
                                             point_size = .2, shader = self.plottingParams['3D.Shader'] )
                
            else:
                self.canvas += self.backend.points( positions = pointPosition, 
                                             color = self.plottingParams['inputMarkerColor'], 
                                             point_size = .1, shader = self.plottingParams['3D.Shader'] )

            offsetChain += self.plottingParams['inputSampleSpacing']
            
            pass # end of per sample for loop
        pass # end of draw_input


    def draw_arch ( self ):
        
        print(':: plotting architecture at timestep ')
        
        nLayers = len( self.archStructure['neuronsInLayers'] )
        self.archStructure['neuronPositions'][self.timeStructure['timeIndex']] = {}
        
        for iLayer in range( nLayers ):
            self.archStructure['neuronPositions'][self.timeStructure['timeIndex']][iLayer] = {}
            
            layerHorizontalOffset = - np.array( [ 0, int( self.archStructure['neuronsInLayers'][iLayer]/2. ) 
                                                     * ( self.plottingParams['neuronSize'] 
                                                        + self.plottingParams['interNeuronHorizontalSpacing']),
                                              0 ] ) 
            layerVerticalOffset = iLayer * self.plottingParams['layerVerticalOffset']
            
            offsetChain = self.plottingParams['offset'] + layerHorizontalOffset + layerVerticalOffset
            
            nNeurons = self.archStructure['neuronsInLayers'][iLayer]
            
            #print( 'plotting layer # ' + str (iLayer) + ' of ' + str(nLayers))

            for iNeuron in range( nNeurons ):
                
                
                self.archStructure['neuronPositions'][self.timeStructure['timeIndex']][iLayer][iNeuron] = offsetChain
                
                #print( 'plotting neuron # ' + str (iNeuron) + ' of ' + str(nNeurons) )
                self.canvas += self.backend.points ( positions = offsetChain, 
                                                      color = self.plottingParams['neuronColor'],
                                                      point_size = self.plottingParams['neuronSize'], 
                                                      shader = self.plottingParams['3D.Shader'])
                
                offsetChain = offsetChain + np.array( [0, self.plottingParams['neuronSize'] 
                                                           + self.plottingParams['interNeuronHorizontalSpacing'], 
                                                       0 ] ) 

    pass # end of draw_arch
    
    def connect_input_to_network ( self ):
        nInputNeurons = len( self.archStructure['neuronPositions'][self.timeStructure['timeIndex']][0] )
        nActiveInputs = self.inputStructure['activeInputPositions'].shape[0]
        if len ( self.inputStructure['activeInputPositions'].shape ) == 1:
            print ('one neuron')


        for iActiveInput in range ( nActiveInputs ):
            for iInputNeuron in range ( nInputNeurons ):
                neuronPos = self.archStructure['neuronPositions'][self.timeStructure['timeIndex']][0][iInputNeuron]
                activeInputPos = self.inputStructure['activeInputPositions'][iActiveInput, :]
                self.canvas += self.backend.line ( (activeInputPos[0], activeInputPos[1], activeInputPos[2],
                                                   neuronPos[0], neuronPos[1], neuronPos[2]), 
                                                      color = self.plottingParams['inputConnectionLineColor'], width = .15 ) 

    def connect_layers_within_timestep ( self ):
        nLayers = len ( self.archStructure['neuronsInLayers'] )
        for iLayer in range( nLayers - 1 ):
            nOriginNeurons = len( self.archStructure['neuronPositions'][self.timeStructure['timeIndex']][iLayer] )
            
            for iOriginNeuron in range ( nOriginNeurons ):
                nDestinationNeurons = len( self.archStructure['neuronPositions'][self.timeStructure['timeIndex']][iLayer + 1] )
                
                for iDestinationNeuron in range ( nDestinationNeurons ):
                    originNeuronPosition = self.archStructure['neuronPositions'][self.timeStructure['timeIndex']][iLayer][iOriginNeuron]
                    destinationNeuronPosition = self.archStructure['neuronPositions'][self.timeStructure['timeIndex']][iLayer + 1][iDestinationNeuron]
                    
                    connectingWeight = self.archStructure['weights'][iLayer][iOriginNeuron][iDestinationNeuron]
                    if connectingWeight > 0:
                        self.canvas += self.backend.line( ( originNeuronPosition[0], originNeuronPosition[1], originNeuronPosition[2],
                                                              destinationNeuronPosition[0], destinationNeuronPosition[1],
                                                               destinationNeuronPosition[2] ), width = 1.5, 
                                                                 color = self.plottingParams['positiveWeightColor'] )
                    else:
                        self.canvas += self.backend.line( ( originNeuronPosition[0], originNeuronPosition[1], originNeuronPosition[2],
                                                           destinationNeuronPosition[0], destinationNeuronPosition[1],
                                                           destinationNeuronPosition[2] ), width = 1.5, 
                                                         color = self.plottingParams['negativeWeightColor'] )
                        
    def connect_to_prev_timestep ( self ):
        
        nLayers = len ( self.archStructure['neuronsInLayers'] )
        
        for iLayer in range( nLayers - 1 ):
            if iLayer in recurrentLayers:
                
                nOriginNeurons = len( self.archStructure['neuronPositions'][self.timeStructure['timeIndex']-1][iLayer] )
                
                for iOriginNeuron in range ( nOriginNeurons ):
                    '''
                    if LSTM_flag:
                        nDestinationNeurons = len( self.archStructure['neuronPositions'][self.timeStructure['timeIndex']][iLayer - 1] ) # LSTM change to account for prev state gate                  
                    else:
                        nDestinationNeurons = len( self.archStructure['neuronPositions'][self.timeStructure['timeIndex']][iLayer] )
                    '''
                    nDestinationNeurons = len( self.archStructure['neuronPositions'][self.timeStructure['timeIndex']][iLayer] )
                    for iDestinationNeuron in range ( min( recurrentNeuronLimit, nDestinationNeurons) ):
                        originNeuronPosition = self.archStructure['neuronPositions'][self.timeStructure['timeIndex']-1][iLayer][iOriginNeuron]
                        '''
                        if LSTM_flag:
                            destinationNeuronPosition = self.archStructure['neuronPositions'][self.timeStructure['timeIndex']][iLayer-1][iDestinationNeuron] # LSTM change                        
                        else:
                            destinationNeuronPosition = self.archStructure['neuronPositions'][self.timeStructure['timeIndex']][iLayer][iDestinationNeuron]
                        '''
                        destinationNeuronPosition = self.archStructure['neuronPositions'][self.timeStructure['timeIndex']][iLayer][iDestinationNeuron]

                        self.canvas += self.backend.line( ( originNeuronPosition[0], originNeuronPosition[1], originNeuronPosition[2],
                                                              destinationNeuronPosition[0], destinationNeuronPosition[1], destinationNeuronPosition[2] ), width = 1, color = self.plottingParams['recurrentWeightColor'] )


class _3DBackend ():
    def __init__ ( self ):
        pass    
    def Draw ( self, drawableObjects ):
        BACKEND_FLAG = 'K3D'
        if BACKEND_FLAG == 'K3D':
            backend += drawableObjects

class NNObject ():
    def __init__ (self, objType, objAttributes):        
        if objType == 'neuron':
            backend.points( positions = objAttributes['positions'],
                               color = objAttributes['positions'],
                               point_size = objAttributes['pointSize'] )