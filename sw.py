import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout

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

#all
x_train, x_test, y_train, y_test = train_test_split(x,y,
                            test_size=0.4,
                             random_state=42,stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#low-level
xL_train, xL_test, y_train, y_test = train_test_split(x_low,y,
                            test_size=0.4,
                             random_state=42,stratify=y)


scaler = StandardScaler()
xL_train = scaler.fit_transform(xL_train)
xL_test = scaler.transform(xL_test)

# Hyperparameters
training_epochs = 1000 # Total number of training epochs
learning_rate = 0.01 # The learning rate


#high-level
xH_train, xH_test, y_train, y_test = train_test_split(x_high,y,
                            test_size=0.4,
                             random_state=42,stratify=y)


scaler = StandardScaler()
xH_train = scaler.fit_transform(xH_train)
xH_test = scaler.transform(xH_test)


#set low-level
model = Sequential()

model.add(Dense(units=300, activation="tanh",input_dim=8))
model.add(Dense(units=2, activation="tanh"))

#model.summary()

#from ann_visualizer.visualize import ann_viz;
#ann_viz(model, view=True, filename="network.gv", title="Shallow Network")

model.compile(optimizer="none", loss='categorical_crossentropy')