# MySw-Comp-

The aim of the code is to find the best algorithm between a Boosted Decision Tree and a Shallow Neural Network to discriminate  singal from background events in the SUSY framework. The data set is the SUSY Data Set taken from the UCI Machine Learning repository, it consits in 5000000 events, each of those characterzied by 8 low-level features and 10 high-level features, these last functions of the first 8.

In the program.ipynb file there is the code in which i prepare the data set, tune the parameters through cross validation for both the BDT and NN and then evaluate the performance exploiting the AUC metric, which is a good one in classification problems, in particular when comparing different algortihms.

I have trained the model using three different datasets: considering all the features, considering just the low-level ones and then just the high-level. This has been done in order to see to what extent the high-level features help the algorithm to learn.
