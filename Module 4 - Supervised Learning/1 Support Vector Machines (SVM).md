#Support Vector Machines (SVM)

<script src="https://code.jquery.com/jquery-3.6.0.min.js" ></script>
<script src="../toc.js" ></script>
<div id='toc'></div>

###Good Reads:
1. [Alexandre KOWALCZYK Book](https://s3.amazonaws.com/ebooks.syncfusion.com/downloads/support_vector_machines_succinctly/support_vector_machines_succinctly.pdf?AWSAccessKeyId=AKIAWH6GYCX3445MQQ5X&Expires=1626440437&Signature=qFtL9LYj0YgPE12IZFrbAPI%2F2E0%3D)
2. [Alexandre KOWALCZYK Math Tutorial](https://www.svm-tutorial.com/2014/11/svm-understanding-math-part-2/)
3. [SVR - Towards data Science](https://towardsdatascience.com/an-introduction-to-support-vector-regression-svr-a3ebc1672c2)

Both **classification** and **regression**.

##Geometric Intution
![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-08 at 5.17.50 PM.png)
![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-08 at 5.24.48 PM.png)

We'll try to find a **hyperplane** $\pi$ such that it separates the group of **points as wide as possible**. Considering the binary classification of $+ve$ and $-ve$ points, we can draw a line $\pi$ in between the 2 groups such that when we draw a parallel line $\pi^{+}$ (positive hyperplane) touching the first point of the +ve group and another line $\pi^{-}$ (negative hyperplane) touching the first point of the -ve group. Then the **margin**, **d** is **dist(**$\pi^+$**,**$\pi^-$**)** and $\pi$ is **margin maximising hyperplane**. The points through which the $\pi^+$ and $\pi^-$ passes are called **support vectors (svs)**.

So, SVM try to find a hyperplane that maximises the margin. This will **minimize the error of misclassification** and **increases the accuracy**.

###Alternate Geometric Intuition
First we'll draw convex-hull. It is a **smallest convex polygon** which covers the external points in a way that all the points are either inside the polygon or on the polygon. And the path between any two points have to be done within the shape and it shouldn't cross the shape.

How to use it?
1. Draw convex hull for the points of each class
2. Draw the shortest line between those hulls
3. Bisect that line to get the $\pi$ (margin maximising hyperplane)

![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-08 at 5.42.57 PM.png)

##Mathematical derivation
###hard margin svm
![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-08 at 5.49.47 PM.png)

But note that $w$ is **not a unit vector** and $w^Tw\neq 0$.

![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-08 at 5.53.54 PM.png)

So, **our problem becomes**

$(w^*,\ b^*)\ =\ \underset{w,b}{argmax}\frac{2}{||w||}$ such that $y_i*(w^Tx_i+b)\geq1$ for all $x_i$s


**But what if the +ve point in -ve region and -ve point in the +ve region (or) not linearly separable (or) almost linearly separable?**
We can't solve the above constraint to solve for $w,b$. So above eqn is **hard margin svm**

---
###soft margin svm
![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-08 at 6.04.48 PM.png)

We'll introduce $\zeta_i$ (zeta i), for all points, such that it's value is zero for points in the proper side of the margin. For others, **the more the point is away from the margin (in the incorrect direction), zeta value increases by some units**. It is the **distance of the point from the correct hyperplane** either $\pi^+$ or $\pi^-$

$\underset{w,b}{argmax}\frac{2}{||w||}$ = $\underset{w,b}{argmin}\frac{||w||}{2}$

$(w^*,\ b^*)\ =\ \underset{w,b}{argmin}\frac{||w||}{2}+c*\frac{1}{n}\sum_{i=1}^n\zeta_i$
such that $y_i*(w^Tx_i+b)\geq1-\zeta_i$ for all $\zeta_i \geq 0$
where $\frac{1}{n}\sum_{i=1}^n\zeta_i$ is the average distance of misclassified points from the correct hyperplane and $c$ is the **hyperparameter**.

Here $\frac{||w||}{2}$ is the **regularization term** and $\frac{1}{n}\sum_{i=1}^n\zeta_i$ is the **loss term**

As $c$ $\uparrow$, we are giving more importance to not make mistakes which leading to **overfit** & **high variance**.
As $c$ $\downarrow$, we are giving less importance to not make mistakes which leading to **underfit** & **high bias**.


It is the **soft margin svm**.

---
###Why we take values +1 and and -1 for Support vector planes

Since we are saying that $||w||\neq1$ and it could be of **any length**.

![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-09 at 10.07.20 AM.png)


![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-09 at 9.50.50 AM.png)
We can take any **k**, as we have the optimization problem only for $w$ as $\frac{2}{||w||}$. So we simply take the value as $k=1$.


##Loss function (Hinge Loss) based interpretation
![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-10 at 4.05.31 PM.png)

Hinge loss is **not differentiable at 0**. But we can handle it.
Hinge loss = $0$ if $z_i \geq 1$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=$\ (1-z_i)$ if $z_i<1$
(OR)
Hinge loss = $max(0, 1-z_i)$

![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-10 at 4.21.51 PM.png)

Consider 2 points,
1. $x_i$ on the correct (positive) side of the $\pi^+$, so $\zeta_i=0$
2. $x_j$ on the wrong side of the $\pi$ (i.e) in the region of $\pi^-$, so the distance of that point from $\pi$ is $w^Tx_j+b$ which is a **-ve** value. What about the distance $d_j$ from $\pi^+$, it is $1-(y_j*(w^Tx_j+b))$ (also $1-z_j$ where $z_j$ is $(y_j*(w^Tx_j+b))$)

<br>
So $\zeta_j\ =\ (1-z_j)$ when $x_j$ is incorrectly classified.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$=\ 0$ when $x_j$ is correctly classified.
It is nothing but $\zeta_j=max(0,1-z_j)$ as seen before.

![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-10 at 4.46.30 PM.png)

Both analysis (soft svm and hinge loss) are conceptually same
![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-10 at 4.49.08 PM.png)
[CORRECTION] : Our constraint is $(1-(y_i(w^x_i+b))\ \leq\ \zeta_i)$

This is the **Primal form**.

## Dual form of SVM formulation
Ref : [https://cs229.stanford.edu/notes2020spring/cs229-notes3.pdf](https://cs229.stanford.edu/notes2020spring/cs229-notes3.pdf)

Another form is **dual form**
$\underset{\alpha_i}{max}\sum_{i=1}^n\alpha_i-\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_j y_i y_jx_i^Tx_j$
such that $C \geq \alpha_i \geq0$ and $\sum_{i=1}^N\alpha_iy_i=0$
Solving **primal form** is equivalent to this **dual form** mathematically.

It is **linear SVM**.

_Observations :_
1. For every $x_i$, we have $\alpha_i$
2. $x_i$ occur in the form of $x_i^Tx_j$. Refers to the **cosine similarity** if ||$x_i$||=1 and ||$x_y$||=1 (i.e) normalized data. We can use the **similarity matrix here** (by substituting the same in the formula). So $x_i^Tx_j$ will be replaced by $k(x_i,x_j)$
3. Usually for any query point $x_q$, we'll find $w^x_q+b\ $ as $f(x_q)$ and take it's sign for the class. Here, we have $f(x_q)$ as $\sum_{i=1}^n\alpha_iy_ix_i^Tx_q+b$
4. $\alpha_i>0$ for SVS (support vector points) and $\alpha_i=0$ for non-SVS (support vector points). This is because we don't care about the points in either side as we can use $f(x_q)$ and find the sign using the SVS points as ref.


## Kernel trick (to use in dual form)

Linear SVM is similar to **logistic regression** as the results don't matter much. But Kernel trick is the most **important idea in SVM**.

Kernel SVM if we replace $x_i^Tx_j$ with $k(x_i,x_j)$. We can classify the **non-linear separable data** with it (like Logistic regression + feature engineering).

###Polynomial Kernel
Generic, $k(x_1,x_2)=(c+x_1^Tx_2)^d$
In quadratic form and with $c$=1, $k(x_1,x_2)=(1+x_1^Tx_2)^2$

![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-10 at 6.00.45 PM.png)
Internally, it is doing **feature transformation in implicit manner**.
![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-10 at 6.02.03 PM.png)
$d'>d$ and data is **linearly separable** (by **Mercers theorem**).

_Question is_ *what kernel to apply ?* It is all about finding right kernel in SVM.

###RBF-Kernel
Radial Basis Function :  Very very general purpose kernel.
$k(x_1,x_2) = exp(\frac{-||x_1-x_2||^2}{2\sigma^2}) = exp(\frac{-d_{12}^2}{2\sigma^2})$ where $\sigma$ is a hyperparameter and $\gamma = \frac{1}{\sigma}$

1. As $d_{12}$ $\uparrow$, $k(x_1,x_2)$ $\downarrow$. It behaves like similarity.
![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-10 at 6.09.23 PM.png)


2. As $d$ increases, kernel value falls to zero (like gaussian PDF).
$\sigma=1$
![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-10 at 6.10.17 PM.png)

$\sigma=0.1$. If d>1, k=0
![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-10 at 6.13.59 PM.png)

$\sigma=10$. If d>10, k=0
![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-10 at 6.15.08 PM.png)


As $\sigma$ increases, we allow more values to be similar. Equivalent to KNN.
![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-10 at 6.18.48 PM.png)

Similar as $\sigma$ $\uparrow$, to $k$ in KNN.
![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-10 at 6.21.07 PM.png)


### Domain Specific Kernels
1. Genome kernel
2. String kernel
3. Graph kernel

Feature transformation is partially replaced by the appropriate kernel.


##Train and run time complexities

We can use $SGD$ algo. But we can use **sequential minimal optimization (SMO)** for **SVM**.

Training time : $O(n^2)$ for kernel SVMs
Runtime : $f(x_q)=\sum_{i=1}^n\alpha_iy_ix_i^Tx_q+b$ It is based on no of vector points. So, complexity is $O(kd)$ where $d$ is the dimensionality of the input and $1\leq k \leq n$.


##nu-SVM: control errors and support vectors
Original SVM is $C-SVM$ (where $C\geq 0$ is a hyperparameter)

Alternate is $nu-SVM$
![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-10 at 6.37.31 PM.png)

where we can control % of errors with $nu$ hyperparam.

if $nu=0.01$, then errors will be $\leq$ $1%$ and $\#\ SVS$ will be $\geq$ $1%$ of N points.


##SVM Regression (SVR)

![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-10 at 6.44.58 PM.png)

$\epsilon$ - hyper parameter
As $\epsilon$ $\uparrow$,  errors will increase and causes **underfit**
As $\epsilon$ $\downarrow$,  errors will be very low in training and causes **overfit**

It can also be kernalised for non-linear data regression.
![](./1 Support Vector Machines (SVM)/Screen Shot 2021-07-10 at 6.45.38 PM.png)


##Cases
1. Feature Engineering and feature transformation (by finding a right kernel)
2. Decision surface is the **non linear surface** for non linear data and **linear surface** after the kernalization.
3. Similarity/distance function can be used by kernels

Challenges
1. We can't find the feature importance or interpretability (directly) for kernel SVMs. But we can use **forward feature selection**
2. Outliers will have very less impact as we'll use only SVS for the kernel SVM.
3. RBF with small $\sigma$ may get affected similar to the smaller **k** in KNN
4. Large **d** - SVM works good

Best Case :
1. Having right kernel, it works well

Worst Case:
1. When **n** is large, Training time is typically long
