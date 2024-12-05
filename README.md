# NeuralNetworkHestonModel
A neural network based calibration method for the Heston model that performs the calibration task at high speeds for the full implied volatility surface. Graphs and charts displaying key results are in the RESULTS folder.


## Key File Breakdown

The HestonAnalytical file contains the analytical implementation for pricing and deducing the implied volatility of European call options under the Heston model. 

The OptionsData file contains code that fetches and organizes data for MSFT options from Yahoo Finance.

The SyntheticHestonData file creates training and testing data for the model via the functions from the HestonAnalytical file

The HestonNeuralNetwork file contains the implentation and training of the neural network that approximates the Heston analytical function

The HestonNNCalibration file then finds the Heston parameters through calibration of this neural network with the MFST option data. Differential evolution is chosen as the method of calibration.
