from keras.models import Sequential
import keras
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from math import sin
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# define X using numpy
X = np.random.random([1000])*2*np.pi-np.pi 

# random funciton Y = f(X)
#Y = X**3 # 
#Y = np.sin(X)+3 # 
#Y = 2*np.sin(X) - np.cos(X) #+ X**4
#mad function below!
Y=[]
for i in X:
	if i>0:
		Y.append(sin(i))
	elif i<-1:
		Y.append(-3)
	else:
		Y.append(i**2)
Y = np.asarray(Y)
# create model
model = Sequential()
model.add(Dense(20, input_dim=1, init='uniform',bias=True, activation='softsign'))
model.add(Dense(1, init='uniform',bias=True, activation='linear'))

# Compile model
model.compile(loss='mse',optimizer='sgd', metrics=['mean_squared_error'])

# Fit the model
model.fit(X, Y, nb_epoch=200, batch_size=1)

 	
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


test = np.random.random([1000])*np.pi*2-np.pi
predict = model.predict(test)
plt.clf()
plt.scatter(X,Y)
plt.scatter(test,predict,color='r')
plt.show()

# best so far: 0.01% with 10 nodes, init uniform, softsign,linear activation


