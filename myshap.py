import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost
import numpy as np
import shap


url = 'pipes_de_stand.csv'
df = pd.read_csv(url)
X, y = df.drop('DE', axis=1), df.DE


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
model = RandomForestRegressor(random_state=10).fit(X_train, y_train)


yhat = model.predict(X_test)
result = mean_squared_error(y_test, yhat)
print('RMSE:',round(np.sqrt(result),4))

explainer = shap.TreeExplainer(model)


shap_values = explainer.shap_values(X_test)
shap_object = explainer(X_train)


#shap.plots.waterfall(shap_values[0])


#shap.summary_plot(shap_values, X_test, feature_names = X_test.columns, plot_type = "bar")

shap.summary_plot(shap_values, X_test);

#shap.plots.beeswarm(shap_object)

