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

## Exploratory Data Analysis (EDA)
In this section we will try to understand the data. We will examine the relationships and dependencies between the variables, evaluate the quality of their linearity, and even try to correct them with mathematical tools

### Box plots
We will scale the data and present them in the boxes. and we can see that most of the data is normal

# image1

###  Transformation for time time variable
We can see the cycle in this variable. Given that the time variable is continuously measured in the range of 0-24, its values do not correctly describe reality linearly, since in the present state, values 0 and 24 are the values with the greatest distance, but in reality they should be the closest. For example: the value of the hour 23:59 is almost 24, and the value of the hour 00:01 is almost 0. The actual difference between them is only 2 minutes

# image2
# image3

To solve this problem, we performed radial transformation using a sinus function
# image4
Let's look at the variables again in vs. serial number of the observation. The color symbolizes y_temp:
# image5
The periodicity of the observations can be seen in each of the parameters. However, in most parameters a linear relationship can be identified with the dependent variable before the coloration of the dots in the top charts, and a tendency to normality according to the density functions of the parameters
# image6

### Correlations
Now we will examine the correlations of the explanatory variables among themselves using a correlation matrix. At this stage we will try to understand how the variables may affect each other. Understanding dependence can help us explain why a statistical model behaves in one form or another and why at different stages of model selection, some of the variables are not statistically significant.
# image7
In this matrix, one can see a high correlation between several variables, which may raise suspicion of multicolinarity between them. Let's look at the charts that represent these relationships:
# image8
In these diagrams we can see the connections between the variables suspected of multicollinearity. Since we have only 9 explanatory variables, we will not drop them now, but we expect that in the model selection process, some of these parameters will drop out.

## Models
In this project I compared linear models and other models of machine learning. In addition, I used statistical methods to select the explanatory variables such as backward selection, and performed various cross validation methods to evaluate the model errors, thus selecting the model structure. I also tried to reduce the dimensionallity of the problem by using PCA transformations and examined the results of the model that uses it.
The comparative models are:
1.	lasso regression
2.	linear regression after backward selection
3.	linear regression after PCA
4.	random forest regression
5.	artificial neural network regression

### Data spliting
I splited the data to train and test. The training set included 65% of the data - 881, and the exam set included 45% of the data-474.
The models were trained using the training set only, which I used for cross validation in each model. Finally I tested all the trained models on the test set to get an indication of the best model.

### Lasso regression
First I ran a scale to the data matrix and then ran 10 folds cross validation. In these graphs we can see the process of validation, error and coefficients in each iteration (α = 0.5)
# image9
The selected lambda is the lambda that minimizes the mean square error 0.004
The model that is attached contains all the parameters:
- (Intercept)   |40.0776582462
- age_sec       | 0.2034198252
- face_max_temp | 2.1724718768
- env_temp      | 1.9985253861
- wall_temp     |-2.2869222491
- quantile1.04  | 0.6196778401
- quantile6.25  |-0.3343201594
- std.30        | 0.1741506208
- std.300       | 0.4250291338
- time_num      |-0.1531332577

 R square= 0.6898, adj R square=0.6867
