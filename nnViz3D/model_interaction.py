def non_singleton_weights ( weights ):
    
    if weights.ndim == 1:
        print('yep')
        return np.expand_dims( weights, axis=1)
    
    else:
        print('nope')
        return weights

def imshow_weights(mParams):
    
    plt.figure()
    
    for iLayer in range(mParams['nLayers']):

        plt.subplot( 1, mParams['nLayers'], iLayer+1 )
        plt.imshow( non_singleton_weights( mParams['weights'][iLayer] ) )
    
    plt.show()
    

def parse_model ( model ):    
    mParams = {}
    mParams['config'] = model.get_config()
    mParams['model'] = Sequential.from_config( mParams['config'] )
    mParams['weights'] = mParams['model'].get_weights()
    mParams['layers'] = mParams['model'].layers; mParams['nLayers'] = len(mParams['layers'])
    mParams['inputs'] = mParams['model'].inputs
    mParams['outputs'] = mParams['model'].outputs
    return mParams