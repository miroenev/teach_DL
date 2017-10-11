import matplotlib.pyplot as plt

def tight_plot(data, colorRGB = (0,0,0,1), legendText = ''):

    plt.figure(figsize=(10,5))
    
    plt.subplots_adjust( left = 0.05, right = 0.95, top = 0.95, bottom = 0.05, wspace = 0.01 )

    plt.plot(data, color = colorRGB)

    plt.autoscale(enable=True, axis='x', tight=True)
    
    if legendText != '':
        plt.legend([legendText], loc='upper right', shadow=True)
    
    plt.grid(linestyle='dotted')