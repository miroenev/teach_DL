
# Training a commute prediction network, and visualizing learning!  
In this notebook we'll visualize the learning/training process of a neural network on a toy problem.

Specifically we'll build a synthetic dataset, sample from it, and try to build a neural network model that predicts data unseen during training. The dataset is intended to capture the relationship between **commute-duration** (i.e., travel time to work, dependent variable) given **time-of-departure** and **weather-condition** (independent variables).

Note that the latest version of this notebook is available from: [https://github.com/miroenev/teach_DL](https://github.com/miroenev/teach_DL). 

Lastly, a video walkthrough of this notebook is <a href='https://youtu.be/HgbGJn9yz30'> available on YouTube</a>.

# Define the problem

Lets try to predict commute duration from two observable independent variables: the time of day and the weather conditions.  

<img src='https://github.com/miroenev/teach_DL/blob/master/figures/commute.png' width='400'/>  

<img src='https://github.com/miroenev/teach_DL/blob/master/figures/target_distribution.PNG' width='1000'/>  

In this toy example we'll first take on the role of the 'traffic gods' and decree that commute duration is defined through a linear mixture of the two independent variables. Later we'll sample from the distribution defined by these variables and generate a training dataset. This sampling procedure will be analogous to keeping a journal of all of our commutes for some [ long ] period of time, where each log entry consists of a set of  
* <b>X</b>: [ time-of-departure, weather-condition ], and the associated  
* <b>Y</b>: [ commute-duration ].

<img src='https://github.com/miroenev/teach_DL/blob/master/figures/x_y_mapping.PNG' width='900'/>  

Given such a journal [dataset], we'll split it into training (75%) and testing (25%) subsets which we'll use to train and evaulate our model respectively. Specifically, we'll build a neural network model whose weights are initially randomly initialized, but are trained/updated as we stream the training data through (via the backpropagation learning algorithm). Each update will get us closer to having a model that has learned the relationship between X and Y or ([ time-of-departure, weather-condition ] to [ commute-duration] ).  

<img src='https://github.com/miroenev/teach_DL/blob/master/figures/process.PNG' width='800'/>  

During the training process we'll try to visualize the network's behavior by asking it to predict all the entries in our logbook using its current parameters/weights. As the training process unfolds, you should be able to see how the network adapts itself to the target surface/function that we determined for the commute duration.  

<img src='https://github.com/miroenev/teach_DL/blob/master/figures/training_progress.PNG' width='700'/>
