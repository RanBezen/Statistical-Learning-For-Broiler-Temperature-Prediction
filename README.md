# Statistical-Learning-For-Broiler-Temperature-Prediction
This project deals with the prediction of broiler temperature using statistical and machine learning. In this project I will predict individual temperature, using various parameters obtained from thermal image.

In this project, I tried to predict temperature of broiler in coop by data from a thermal camera. The data came from an experiment conducted in a broiler coop that included precise sensors and cameras. In this experiment, 1355 observations were collected, 65% of which were used to train the models, and 35% were used to test the models.
I made predictions in several different statistical methods. First I examined the data, exploratory data analysis, tried to understand how each variable might explain the problem, and examined the relationships and dependencies between the variables. I used linear models such as linear regression and lasso regression, used statistical methods to select the explanatory variables such as backward regression. I also performed various cross validation methods to evaluate the error and select a model, I have also tried to reduce the problem dimensions with Transformations of  Principal Components Analysis (PCA).

I evalute the three Liniar models and saw that their test scores were lower than the results I wish for. So, I decided to try other methods of machine learning. I tried two methods: Random Forest and Artificial Neural Networks. In each model I performed cross validation in order to tune the parameters: in Random Forest these were the number of trees and the amount of parameters in each tree, and in these Artificial Neural Networks were the number of neurons in each layer.

The models were estimated using three indicates: Rsquare, root mean square error (RMSE), and an absolute absolute error rate (MAPE). Using these measures I could compare different regression models (linear and nonlinear). The results showed that the Artificial Neural Networks model had the best scores with R square = 0.8121, RMSE = 0.2451, MAPE = 2.26%

* The project is part of a commercial product development, so I can not share the full data


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

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/1.png)

###  Transformation for time time variable
We can see the cycle in this variable. Given that the time variable is continuously measured in the range of 0-24, its values do not correctly describe reality linearly, since in the present state, values 0 and 24 are the values with the greatest distance, but in reality they should be the closest. For example: the value of the hour 23:59 is almost 24, and the value of the hour 00:01 is almost 0. The actual difference between them is only 2 minutes

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/2.png)
![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/3.png)

To solve this problem, we performed radial transformation using a sinus function

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/4.PNG)

Let's look at the variables again in vs. serial number of the observation. The color symbolizes y_temp:

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/5.png)

The periodicity of the observations can be seen in each of the parameters. However, the most of parameters has a linear relationship with the dependent variable by the coloration of the dots in the top charts, and a tendency to normality according to the density functions of the parameters.

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/6.png)

### Correlations
Now we will examine the correlations of the explanatory variables among themselves using a correlation matrix. At this stage we will try to understand how the variables may affect each other. Understanding dependence can help us explain why a statistical model behaves in one form or another and why at different stages of model selection, some of the variables are not statistically significant.

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/7.png)

In this matrix, one can see a high correlation between several variables, which may raise suspicion of multicolinarity between them. Let's look at the charts that represent these relationships:

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/8.PNG)

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

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/9.PNG)

The selected lambda is the lambda that minimizes the mean square error 0.004
The model that is attached contains all the parameters:
- (Intercept)   | 40.0776582462
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
 
 You can see the residual chart and the QQplot:
We can identify a normal distribution of errors and equality of variances around the 0.
The Shapiro Wicks test is significant: W = 0.99497747, p-value = 0.005289821

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/10.png)

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/11.png)

### backward regression
Here I did leave one out Cross Validation on a backward selection algorithm to select the best model.
The selected model contains 6 parameters and  an intercept, it can be seen that the explanatory variables are the standard deviation of 30 px and the two percentiles. They were probably scaled because of multicolinarity as we suspected in the EDA 

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/12.png)

Rsqr=0.6879, adj R sqr=0.6858

You can see the residual chart and the QQplot:
We can identify a normal distribution of errors and equality of variances around the 0.
The Shapiro Wicks test is significant: W = 0.99497429, p-value = 0.005266207

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/13.png)

### Principal Component Analysis (PCA) regression 
With PCA I took all data. My assumption was that linear transformations could be performed by using the eigan vectors of the highest eigan values in the covariance matrix to reduce the dimensionallity, but would still explain 80% of the variance of the data.
It is possible to see that the first three main factors (PC1, PC2, PC3) are sufficient to explain 88.49% of the variance.

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/14.png)

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/15.PNG)

It can be seen that with this method, the dimension of the problem decreases significantly, but the regression indices decrease too:

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/16.png)

Rsqr=0.57, adj R sqr=0.5685
 
  You can see the residual chart and the QQplot:
We can identify a normal distribution of errors and equality of variances around the 0.
The Shapiro Wicks test is significant: W = 0.99437565, p-value = 0.002290684

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/17.png)

### Random forest
In this model I chose two parameters: the number of parameters in each tree, and the number of trees. I made the choice with 10 folds cross validation and with grid search. Finally, I took the model with the smallest mean square error.
In these graphs you can see the grid search optimization process. Each diagram shows one free parameter against the MSE, given that the second parameter is fixed on its optimal value

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/18.PNG)

At the end of the parameter adjustment process, the lowest MSE value was 43 trees with 7 parameters per tree

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/19.PNG)

Training error graph

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/20.png)

### Artificial neural networks with 3 hidden layers
In this model I chose 3 parameters that represent the number of neurons in each hidden layer. I made the choice with 10 folds cross validation by using a grid search of the three parameters to minimize the mean square error estimated.

before the network training, I made a scaled the data (both x and y) to easier the network weights to convergence.


In the following diagrams, the MSE vs. number of neurons in one layer, while the number of neurons in the other two layers is determined by the optimal value found

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/21.png)

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/22.png)

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/23.png)
Finally, the model with the smallest mean square error is a model with 5 neurons in the first layer, 5 neurons in the second layer, and 3 neurons in the third layer. validation mean square error: 0.0567
The final structure of the network:

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/24.png)

## Results
After the training, I tested the models on test data set. The indices I used for comparison are Rsquare, root mean square error (RMSE) and Mean absolute percent error (MAPE). These three measures can compare linear and nonlinear regression models. 

test results:

|     model    |       R square     |   RMSE     |  MAPE    |
| ------------- | ------------- |------------|------------|
| Lasso|	0.6843|	0.3185|	2.98|
| Backward|	0.6947	|0.3115|	2.91|
| PCA	|0.5566	|0.3753	|3.50|
| Random Forest	|0.7984	|0.2531	|2.31|
| Neural Networks|	0.8121|	0.2451	|2.26|

We can see that the neuronal networks have the best results in the three measures:
R square=0.8121 , RMSE=0.2451, MAPE =2.26%


The following graphs show the test results. The red line marks the RMSE and green the Rsqaure. In the bottom chart you can see the MAPE marked on the blue line

![Image description](https://github.com/RanBezen/Statistical-Learning-For-Broiler-Temperature-Prediction/blob/master/images/25.PNG)


