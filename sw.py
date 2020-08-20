import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
import joypy

names=["class label", "lepton 1 pT", "lepton 1 eta", "lepton 1 phi",
                     "lepton 2 pT", "lepton 2 eta", "lepton 2 phi",
                     "missing energy magnitude", "missing energy phi", 
                     "MET_rel", "axial MET","M_R", "M_TR_2", "R", "MT2", "S_R",
                     "M_Delta_R", "dPhi_r_b", "cos(theta_r1)"]
df = pd.read_csv('SUSY.csv', 
                 encoding='utf-8', 
                 comment='#',
                 sep=',',
                 names=names)
#print(df.head(10))


#plt.scatter(df["lepton 1 pT"], df["lepton 1 eta"])      
#plt.show()



#print(df.keys())

#print('Example number, feature number:', df.shape)


#npart = ["lepton 1 pT", "lepton 1 eta", "lepton 1 phi"]
#for key in npart:
#   matplotlib.rcParams.update({'font.size': 16})
#  fig = plt.figure(figsize=(8,8), dpi=100)
    
    # -- declare common binning strategy (otherwise every histogram will have its own binning)
#   bins = np.linspace(min(df[key]), max(df[key]) + 1, 30)
    
    # plot!
#   _ = plt.hist(df[key], histtype='step', bins=bins, label=r'$t\overline{t}$', linewidth=2)
#  plt.show()

#I want to study three different model of classification: with just low-level, with just high-level and with both of them. Hence i'll define different arrays.


#All features
x=df.iloc[:,1:]
x = x.values

#low-level features
x_low=df.iloc[:,1:9]
x_low = x_low.values

#high-level features
x_high=df.iloc[:,10:]
x_high = x_high.values

#target
y = df["class label"].values

plt.figure(figsize=[70, 56])

""" fig, axL = plt.subplots(4,2)
axL[0,0].hist(x[:,1][y==0], density=True,histtype='step', bins=70, label='Background')
axL[0,0].hist(x[:,1][y==1], density=True,histtype='step', bins=70, label='Signal')
axL[0,0].set_title("pT1 sig and bkg", fontsize = 16, color = 'black', alpha = .5)                        
axL[0,0].set_xlabel('pT',  fontsize = 16, color = 'black', alpha = 1)


axL[0,1].hist(x[:,2][y==0], density=True,histtype='step', bins=70, label='Background')
axL[0,1].hist(x[:,2][y==1], density=True,histtype='step', bins=70, label='Signal')
axL[0,1].set_title("eta1 sig and bkg", fontsize = 16, color = 'black', alpha = .5)                        
axL[0,1].set_xlabel('eta',  fontsize = 16, color = 'black', alpha = 1)

axL[1,0].hist(x[:,3][y==0], density=True,histtype='step', bins=70, label='Background')
axL[1,0].hist(x[:,3][y==1], density=True,histtype='step', bins=70, label='Signal')
axL[1,0].set_title("phi1 sig and bkg", fontsize = 16, color = 'black', alpha = .5)                        
axL[1,0].set_xlabel('phi',  fontsize = 16, color = 'black', alpha = 1)

axL[1,1].hist(x[:,4][y==0], density=True,histtype='step', bins=70, label='Background')
axL[1,1].hist(x[:,4][y==1], density=True,histtype='step', bins=70, label='Signal')
axL[1,1].set_title("pT2 sig and bkg", fontsize = 16, color = 'black', alpha = .5)                        
axL[1,1].set_xlabel('pT',  fontsize = 16, color = 'black', alpha = 1)

axL[2,0].hist(x[:,5][y==0], density=True,histtype='step', bins=70, label='Background')
axL[2,0].hist(x[:,5][y==1], density=True,histtype='step', bins=70, label='Signal')
axL[2,0].set_title("eta2 sig and bkg", fontsize = 16, color = 'black', alpha = .5)                        
axL[2,0].set_xlabel('eta',  fontsize = 16, color = 'black', alpha = 1)


axL[2,1].hist(x[:,6][y==0], density=True,histtype='step', bins=70, label='Background')
axL[2,1].hist(x[:,6][y==1], density=True,histtype='step', bins=70, label='Signal')
axL[2,1].set_title("phi2 sig and bkg", fontsize = 16, color = 'black', alpha = .5)                        
axL[2,1].set_xlabel('phi',  fontsize = 16, color = 'black', alpha = 1)

axL[3,0].hist(x[:,7][y==0], density=True,histtype='step', bins=70, label='Background')
axL[3,0].hist(x[:,7][y==1], density=True,histtype='step', bins=70, label='Signal')
axL[3,0].set_title("missing energy magnitude sig and bkg", fontsize = 16, color = 'black', alpha = .5)                        
axL[3,0].set_xlabel('E',  fontsize = 16, color = 'black', alpha = 1)

axL[3,1].hist(x[:,8][y==0], density=True,histtype='step', bins=70, label='Background')
axL[3,1].hist(x[:,8][y==1], density=True,histtype='step', bins=70, label='Signal')
axL[3,1].set_title("missing energy phi sig and bkg", fontsize = 16, color = 'black', alpha = .5)                        
axL[3,1].set_xlabel('phi',  fontsize = 16, color = 'black', alpha = 1) """


#plt.show()


class_counts= df.groupby('class label').size()
#print(class_counts)



#all
x_train, x_test, y_train, y_test = train_test_split(x,y,
                            test_size=0.4,
                             random_state=42,stratify=y)



#low-level
xL_train, xL_test, y_train, y_test = train_test_split(x_low,y,
                            test_size=0.4,
                             random_state=42,stratify=y)





#high-level
xH_train, xH_test, y_train, y_test = train_test_split(x_high,y,
                            test_size=0.4,
                             random_state=42,stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

xL_train = scaler.fit_transform(xL_train)
xL_test = scaler.transform(xL_test)

xH_train = scaler.fit_transform(xH_train)
xH_test = scaler.transform(xH_test)          

import xgboost as xgb

#i will use early stopping in order to see the ideal number of n_rounds. 
# After early_stopping_rounds without improvements, the train will stop
dtrain = xgb.DMatrix(x_train,y_train)
dtest = xgb.DMatrix(x_test,y_test)


""" 
evallist = [(dtest, 'eval'), (dtrain, 'train')]
param = {'max_depth': 9, 'eta': 0.2, "min_child_weight": 7}
param['objective'] ='binary:logistic' #good for classification
param['eval_metric'] = "auc" #auc,rmse,roc. This evaluate how good the model is. AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.
num_round = 70 #low eta means larger num_round
bst = xgb.train(param, dtrain, num_round, evallist,early_stopping_rounds=6)

print("Best AUC: {:.3f} with {} rounds".format(
                 bst.best_score,
                 bst.best_iteration+1)) """
# make prediction
#preds = bst.predict(dtest)

# print accuracy score
#print(np.round(accuracy_score(y_test, preds)*100, 2), '%')

#In order to tune the other hyperparameters, we will use the cv function to run cross-validation on our training dataset 



""" CV MAX DEPTH AND CHILD WEIGHT

# I need to find the best parameters in order
# to have the best model which minimize the error
# to do that i perform cv trying different values of parameters
# in selected ranges.
# i start searching for max_depth, min_child_weight which helps in fixing the complexity and controlling
# overfit of the model.
param = { }
param['objective'] ='binary:logistic' #good for classification
param['eval_metric'] = "error"
num_round = 70

gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(8,11) # i saw that 15 makes it to overfit, after some rounds the error starts to increase
    for min_child_weight in range(6,8)
]
# Define initial best params and error
min_err = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
# Update our parameters
    param['max_depth'] = max_depth
    param['min_child_weight'] = min_child_weight
# Run CV
    cv_results = xgb.cv(
        param,
        dtrain,
        num_boost_round=num_round,
        seed=42,
        nfold=3, # 5 was too long, choose this beacause the sample is large and it takes too much time
        metrics={'error'},
        early_stopping_rounds=6
    )
# Update best error
    mean_err = cv_results['test-error-mean'].min()
    boost_rounds = cv_results['test-error-mean'].argmin()
    print("\tError: {} for {} rounds".format(mean_err, boost_rounds))
    if mean_err < min_err:
        min_err = mean_err
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, Error: {}".format(best_params[0], best_params[1], min_err)) """



min_err = float("Inf")
best_params = None

for eta in [.3, .2, .1, .05]:
    print("CV with eta={}".format(eta))
    # We update our parameters
    param={}
    param['objective'] ='binary:logistic' #good for classification
    param['eval_metric'] = "error"
    param['max_depth'] = 9
    param['min_child_weight'] = 6
    param['eta'] = eta
    # Run and time CV
    cv_results = xgb.cv(
            param,
            dtrain,
            num_boost_round=70,
            seed=42,
            nfold=3,
            metrics=['error'],
            early_stopping_rounds=6
          )
    # Update best score
    mean_err = cv_results['test-error-mean'].min()
    boost_rounds = cv_results['test-error-mean'].argmin()
    print("\tError {} for {} rounds\n".format(mean_err, boost_rounds))
    if mean_err < min_err:
        min_err = mean_err
        best_params = eta
print("Best params: {}, MAE: {}".format(best_params, min_err))






""" NEURAL NETWORK
from keras.models import Sequential
from keras.layers import Dense, Dropout 
# Hyperparameters
#training_epochs = 1000 # Total number of training epochs
#learning_rate = 0.01 # The learning rate

#set low-level
#model = Sequential()

#model.add(Dense(units=300, activation="tanh",input_dim=8))
#model.add(Dense(units=2, activation="tanh"))

#model.summary()

#from ann_visualizer.visualize import ann_viz;
#ann_viz(model, view=True, filename="network.gv", title="Shallow Network")

#model.compile(optimizer="none", loss='categorical_crossentropy')"""