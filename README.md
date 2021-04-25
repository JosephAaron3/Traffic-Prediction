# Traffic Prediction
Implementing several neural networks described in the paper by Mena-Oreja and Gozalvez (2020) to predict near-future traffic dynamics attributes along Californian highways.

*Source paper:* J. Mena-Oreja and J. Gozalvez, "A Comprehensive Evaluation of Deep Learning-Based Techniques for Traffic Prediction," in IEEE Access, vol. 8, pp. 91188-91212, 2020, doi: 10.1109/ACCESS.2020.2994415

## Data:
- California Department of Transport PeMS real-time traffic data
	- Downloaded as a historical dataset from https://github.com/susomena/pems-traffic-prediction-datasets
	- Indexed by time (5 minute intervals) and space (traffic detector # along a highway)
	- 3 attributes for traffic dynamics - flow, density and speed
	- Approx. ~17 million spatiotemporal observations (2 years, 3 highways)

The diagram below shows an illustration of the data. The vertical axis is time, the horizontal axis is detector number, and the lined circles are detector instances. Although the data will need to be transformed according to the deep learning model architecture used, the illustration demonstrates a situation where, at a certain time (blue), we might want to predict traffic behaviour 15 minutes in the future (pink) using traffic behaviour from the previous 30 minutes (green). Typically we only predict one attribute, using all 3 attributes to train the network.

![data_layout](/Output/Untitled%20Diagram.png)

## Models:
*Notation:*
- T = Number of time periods in the past from which to predict (e.g. T = 12 => 60 minutes of historical observations)
- D = Number of detectors
- P = Number of predictors (1-3)
- Batch size is not specified below for simplicity

### MLP
This is a basic multi-layer perceptron with 2 fully-connected layers (using leakyReLU) and a 60% dropout in between. Input tensor has dimension (D, T, P).

**Architecture:**


**Performance:**

### CNN
Convolutional model with 9 residual blocks (2D Convolutional layers for space-time dimensions + residual connections) followed by 2 dense layers. Input tensor has dimension (D, T, P).

**Architecture:**


**Performance:**


### LSTM
Simple long short-term memory RNN with x40 memory cells. Input tensor has dimension (T, P).

**Architecture:**


**Performance:**


### CNN_LSTM
A mixture of the previous 2 models, without residuals, and only convolving in the spatial dimension. Input tensor has dimension (D, P).

**Architecture:**


**Performance:**
