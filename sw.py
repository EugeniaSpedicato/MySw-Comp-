import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


fig, axL = plt.subplots(4,2)
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
axL[3,1].set_xlabel('phi',  fontsize = 16, color = 'black', alpha = 1)

#plt.show()

""" axes[0,1]
axes[1,0]
axes[1,1]
axes[2,0]
axes[2,1]
axes[3,0]
axes[3,1]
4,2,
                          column=["lepton 1 pT", "lepton 1 eta", "lepton 1 phi",
                        "lepton 2 pT", "lepton 2 eta", "lepton 2 phi",
                        "missing energy magnitude", "missing energy phi"],
                          by = "class label", # separate the data by this value. Creates a separate distribution for each one.
                          ylim = 'own', 
                          figsize = (12,8), 
                          legend = True, 
                          color = ['#f4cccc', '#0c343d'], 
                          alpha = 0.4 """






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


from sklearn.tree import DecisionTreeClassifier

#I need to find the best depth for the tree hence i first leav it default such that the model will decide it. Then i try with others comparing with the results on the test set to understand if it is over or under fitting.

tree_all= DecisionTreeClassifier()
tree_all.fit(x_train, y_train)
print(tree_all)
y_pred_test = tree_all.predict_proba(x_test)
print(y_pred_test)














""" 
from keras.models import Sequential
from keras.layers import Dense, Dropout 
#scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
scaler = StandardScaler()
xL_train = scaler.fit_transform(xL_train)
xL_test = scaler.transform(xL_test)
scaler = StandardScaler()
xH_train = scaler.fit_transform(xH_train)
xH_test = scaler.transform(xH_test)
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