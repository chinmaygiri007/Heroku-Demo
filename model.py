#Random_Forest_Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv("Position_Salaries.csv")
X = data["Level"].values.reshape(-1,1)
Y = data["Salary"].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,random_state = 0)
regressor.fit(X,Y)

Y_pred = regressor.predict(X)

#Saving model to disk
pickle.dump(regressor,open("model.pkl","wb"))

#Loading model to compare the result
model = pickle.load(open("model.pkl","rb"))
print(model.predict([[6.5]]))

X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color = "gray")
plt.plot(X_grid,regressor.predict(X_grid),color = "black")
plt.title("Position Vs Salaries")
plt.xlabel("Level")
plt.ylabel("Salaries")
plt.show()