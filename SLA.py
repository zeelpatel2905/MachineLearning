import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Salary_Data.csv")
df=pd.DataFrame(dataset)

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=15)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

pred_test=regressor.predict(x_test)
pred_train=regressor.predict(x_train)

#train
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,pred_train,color="green")
plt.title("Train")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#test
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_train,pred_train,color="green")
plt.title("Test")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()



