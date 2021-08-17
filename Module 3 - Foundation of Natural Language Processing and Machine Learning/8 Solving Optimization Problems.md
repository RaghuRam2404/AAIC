# Solving Optimiation Problems
<script src="https://code.jquery.com/jquery-3.6.0.min.js" ></script>
<script src="../toc.js" ></script>
<div id='toc'></div>


We have seen it till now for PCA, linear and logistic regression. We'll use the basics of differentiation (scalar and vector) and concepts of maxima & minima.


## Single Variable Calculus (Scalar)

![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 6.20.50 PM.png)

![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 6.25.43 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 6.27.40 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 7.38.13 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 7.39.21 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 7.40.23 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 7.40.42 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 7.41.31 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 7.42.36 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 7.43.16 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 7.45.45 PM.png)

## Maxima and Minima

**At minima and maxima, the slope becomes zero.**
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 7.56.33 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 7.58.33 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 7.59.53 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.00.17 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.01.56 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.03.28 PM.png)

##Vector calculus: Grad
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.12.04 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.22.41 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.24.21 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.27.01 PM.png)
Solving this is very hard. So we'll use **Gradient Descent** to solve it without solving for $\frac{d(f(x))}{dx}=0$.

##Gradient descent: geometric intuition
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.29.41 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.32.06 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.33.11 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.36.03 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.37.35 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.38.25 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.39.06 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.40.29 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.41.46 PM.png)

##Learning rate
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.42.41 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.44.48 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.46.44 PM.png)

##Gradient descent for linear regression
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.49.41 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.50.38 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.51.54 PM.png)
There is a problem with this in case of **large dataset**.
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.52.59 PM.png)

##SGD algorithm
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.56.50 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 8.58.24 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 9.00.05 PM.png)
Correction : $k << n$

##Constrained Optimization & PCA
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 9.02.44 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 9.03.56 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 9.07.14 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 9.30.56 PM.png)
Correction : $S=Cov(X) =\frac{X_T*X}{n}$
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 9.32.22 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 9.33.07 PM.png)

##Logistic regression formulation revisited
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 9.37.37 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 9.39.16 PM.png)

##Why L1 regularization creates sparsity?
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 9.41.39 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 9.44.03 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 9.46.06 PM.png)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 9.50.03 PM.png)
The speed at which the $-r*(2*w_{ij})$ is small for $L2$ but constant $1$ for $L1$ (making sparsity for L1 reg)
![](./8 Solving Optimization Problems/Screen Shot 2021-07-07 at 9.51.14 PM.png)

