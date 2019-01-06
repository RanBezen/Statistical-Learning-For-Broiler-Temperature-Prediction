# Statistical-Learning-For-Broiler-Temperature-Prediction
This project deals with the prediction of broiler temperature using statistical and machine learning. In this project I will predict individual temperature, using various parameters obtained from thermal image.

In this project, I tried to predict temperature of broiler in coop by data from a thermal camera. The data came from an experiment conducted in a broiler coop that included precise sensors and cameras. In this experiment, 1355 observations were collected, 65% of which were used to train the models, and 35% were used to test the models.
I made predictions in several different statistical methods. First I examined the data, exploratory data analysis, tried to understand how each variable might explain the problem, and examined the relationships and dependencies between the variables. I used linear models such as linear regression and lasso regression, used statistical methods to select the explanatory variables such as backward regression. I also performed various cross validation methods to evaluate the error and select a model, I have also tried to reduce the problem dimensions with Transformations of Primary Factor Analysis (PCA).
I evalute the three Liniar models and saw that their test scores were lower than the results I wish for. So, I decided to try other methods of machine learning. I tried two methods: Random Forest and Artificial Neural Networks. In each model I performed cross validation in order to tune the parameters: in Random Forest these were the number of trees and the amount of parameters in each tree, and in these Artificial Neural Networks were the number of neurons in each layer.

The models were estimated using three indicates: Rsquare, root mean square error (RMSE), and an absolute absolute error rate (MAPE). Using these measures I could compare different regression models (linear and nonlinear). The results showed that the Artificial Neural Networks model had the best scores with R square = 0.8121, RMSE = 0.2451, MAPE = 2.26%

## Data structure

|     Params    |       Type     |   Mean     |  sd    |
| ------------- | ------------- |------------|------------|
| Y_temp | continuous   |   41.26     |     0.54     |
| face_max_temp  |continuous   | 37.79  |     0.71     |
| Age_sec (in seconds)  |continuous |270207.17  |129798.48     |
| env_temp |continuous |28.02  |2.03     |
| Wall_temp |continuous |28.35 |1.87     |
| quantile1.04 |continuous | 36.45 |0.87     |
| quantile6.25 |continuous |30.99 |1.84     |
| std.30 |continuous |0.22 |0.06     |
| std.300 |continuous   | 2.05 |0.5     |
| time |continuous |12.17|6.5|
