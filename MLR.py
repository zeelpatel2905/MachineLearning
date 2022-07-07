import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds=pd.read_csv("Invest2Profit.csv")
df=pd.DataFrame(ds)

x=df.iloc[:,:-1].values
y=df.iloc[:,4].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# labelencode=LabelEncoder()
# x[:,3]=labelencode.fit_transform(x[:,3])
# hot=OneHotEncoder(categorical_features=4)
# x=hot.fit_transform(x).toarray()
ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[3])],remainder="passthrough")
x=np.array(ct.fit_transform(x))
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

pred_test=regressor.predict(x_test)
pred_train=regressor.predict(x_train)

print("Train Score:",regressor.score(x_train,y_train))
print("Test Score:",regressor.score(x_test,y_test))



