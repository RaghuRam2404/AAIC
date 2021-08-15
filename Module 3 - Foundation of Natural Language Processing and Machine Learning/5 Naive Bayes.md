# Naive Bayes

<script src="https://code.jquery.com/jquery-3.6.0.min.js" ></script>
<script src="../toc.js" ></script>
<div id='toc'></div>

## Probability ideas

### Conditional Probability

A, B are random variables.
$P(A|B)\ =\ Pr(A=a|B=b)$ - what is the probability of A given B occurs?
$P(A|B)=\frac{P(A\bigcap B)}{P(B)}$ only if $P(B)!=0$

Throwing two dices $D_1$ annd $D_2$, we have 36 possible outcomes. $P(D_1=2)=6/36=1/6$

What is $P(D_1+D_2\leq 5)$? It is $10/36$
Then what is $P(D_1=2\ |\ D_1+D_2\leq 5)$, it is $\frac{3/36}{10/36} = 3/10=0.3$

##Independent vs Mutually exclusive events

###Independent
2 events $A$ & $B$ are said to be independent if 
$P(A|B)=P(A)$
$P(B|A)=P(B)$

$A$ : Getting 6 in Dice 1
$B$ : Getting 3 in Dice 2

$P(A=6|B=3)=P(A=6)$ because B won't influence the A's outcome
$P(B=3|A=6)=P(B=3)$ because A won't influence the B's outcome

### Mutually Exclusive

A : Dice 1 getting value of 6
B : Dice 1 getting value of 3

$P(A|B)=P(B|A)=0$ because $P(A\bigcap B)=0$ because we can't get 2 output in a single dice.

## Bayes Theorem

$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$ givenn $P(B)\neq0$

$P(A|B)\ \longrightarrow$ posterior probability
$P(B|A)\ \longrightarrow$ likelihood probability
$P(A)\ \longrightarrow$ prior
$P(B)\ \longrightarrow$ evidence

_Ex:_
20% of items are produced in Machine $A_1$, 30% of items are produced in Machine $A_2$ and 50% of item in $A_3$ in a factory. Their corresponding defect rate is 5%, 3%, 1%. Picking a random item from the entire items and it is found to be defective, what is the probability that this item is from third machine?

_Ans:_
$P(A_1)=0.2$, $P(A_2)=0.3$, $P(A_3)=0.5$

B : Probability of item being defective
$P(B|A_1)=0.05$, $P(B|A_2)=0.03$, $P(B|A_3)=0.01$

we need to find $P(A_3|B)$ (i.e) Probability of getting an item from third machine given that it is defective.

$P(B)$ = $\sum_{i=1}^3P(B\bigcap A_i)$ = $\sum_{i=1}^{3}P(B|A_i)P(A_i)$ = (0.2\*0.05) + (0.3\*0.03) + (0.5\*0.01) = 0.024

2.4% is the probability of item being defective from 3 machines.

Now, $P(A_3|B) = \frac{P(B|A_3)P(A_3)}{P(B)}$ = $\frac{0.5*0.01}{0.024}$ = $0.2083$

For curiosity, let's find what is probability of an defective item from $A_1$ and $A_2$,
$P(A_1|B) = \frac{P(B|A_1)P(A_1)}{P(B)}$ = $0.4167$
$P(A_2|B) = \frac{P(B|A_2)P(A_2)}{P(B)}$ = $0.375$

$P(A_2|B)+P(A_2|B)+P(A_3|B)=1\ \longrightarrow$ probability of an defective item from all machines

## Naive Bayes algorithm

It is based on **probability**.

$x=(x_1, x_2\ ...\ x_n)$ with $n$ features

$P(C_k\ |\ x_1, x_2\ ...\ x_n)$ for each of $k$ possible outcomes or classes $C_k$

$P(C_k|x) = \frac{P(C_k)P(x|C_k)}{P(x)}= \frac{P(C_k,x)}{P(x)}$
$P(C_k,x)$ = $P(C_k,x_1,x_2 ... x-n)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= $P(x_1|x_2 ... x_n, C_k)P(x_2 ... x_n, C_k)$ (by chain rule)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= $P(x_1|x_2 ... x_n, C_k)P(x_2 | x_3, x_4 ... x_n, C_k)P(x_3, x_4, .. x_n, C_k)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= ...
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= $P(x_1|x_2 ... x_n, C_k)P(x_2 | x_3, x_4 ... x_n, C_k)\ ..... \ P(x_{n-1}| x_n, C_k)P(x_n|C_k)P(C_k)$

$P(x_1|x_2 ... x_n, C_k)$ calculating this term is very difficult as we need to exact $x$ & $C_k$ values in our data set.

Independence : $P(A|B) = P(A)$
Conditional Independence : $P(A|B,C) = P(A|C)$ A&B are conditionally independent given C

So we can say that $x_i$ is **conditionally independent** of $x_{i+1}, x_{i+2}\ ...\ x_n$ given $C_k$ (i.e)
$P(x_i\ |\ x_{i+1}, x_{i+2}\ ...\ x_n, C_k) = P(x_i|C_k)$
*_Naive part : the conditional independent of the variables given_* $C_k$

$P(C_k,x)= P(x_1|x_2 ... x_n, C_k)P(x_2 | x_3, x_4 ... x_n, C_k)\ ..... \ P(x_{n-1}| x_n, C_k)P(x_n|C_k)P(C_k)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\propto P(x_1|C_k)\ P(x_2|C_k)\ P(x_3|C_k)\ ... P(x_{n-1}|C_k)P(x_n|C_k)P(C_k)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\propto P(C_k)\prod_{i=1}^nP(x_i|C_k)$

###maximum of posteriori rule
$\hat{y}=\underset{k\ \epsilon\ {1,2...k}}{argmax}\ p(c_k)\prod_{i=1}^n\ p(x_i,c_k)$

Works well for **categorical features**.

###Implementation for categorical features
1. Find the conditional probability of each $d$ features $P(x_i|C_k)$ in each class $c$ for all $nn$ points
2. For testing, query for the conditional probability and multiple all of them
3. Take argmax for the class

Check this [http://shatterline.com/blog/2013/09/12/not-so-naive-classification-with-the-naive-bayes-classifier/](http://shatterline.com/blog/2013/09/12/not-so-naive-classification-with-the-naive-bayes-classifier/)

####Space & time complexity
**_For training :_**
Simple brute force implementation's **time complexity** is $O(ndc)$ where $n$ is no of points, $d$ is the dimension and $c$ classes.
**Space complexity** after training is O(dc)
**_For testing:_**
**time complexity** is $O(dc)$

## Naive Bayes on Text data
Popular in text classification. Like **Spam filter**, **polarity of the review**.

Say we have set of sentences/text and it's corresponding class. We'll do the removal of stop words, stemming, lemmatization first. We'll be end up with set of words. Binary BoW is good for **spam filter**.

$y\ \epsilon\ \{0,1\}$
$text\longrightarrow \{w_1, w_2, w_3 .... w_d\}$ 

$P(y=1|text)\ =\ P(y=1|w_1,w_2,...w_d)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ \propto P(y=1)P(w_1|y=1)P(w_2|y=1)...P(w_d|y=1)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\propto P(y=1)\prod_{i=1}^dP(w_i|y=1)$

We can calculate $P(y=1)$ and $P(y=0)$
$P(w_i|y=1) = \frac{no.\ of\ datapoints\ contain\ w_i\ with\ class\ label\ y=1}{no.\ of\ datapoints\ with\ class\ label\ y=1}$

This will act as **benchmark** for other classification problems.

###Laplace/Additive Smoothing

In test data, we have $text_q=\{w_1, w_2, w_3, w'\}$
We have $\{w_1, w_2, w_3\}$ in our training data and we have conditional probabilitis for those, but not $w'$.

$P(y=1|text_q) = P(y=1|w_1,w_2,w_3,w')$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$=P(y=1)*P(w_1|y=1)*P(w_2|y=1)*P(w_3|y=1)*P(w'|y=1)$

What is the value of $P(w'|y=1)$ and $P(w'|y=1)$? We can't ignore $w'$ word's conditional probability.

By definition, $P(w'|y=1)$ = $\frac{P(w1,y=1)}{P(y=1)}$ = $\frac{0}{n_1}$ = 0

We can do **laplace/additive smoothing** now.

$P(w_i|y=1)= \frac{\#\ of\ occurences\ of\ w_i\ in\ corpus\ +\ \alpha}{n_1+\alpha k}$ 
where $n_1$ is the no of data points in the particular class of $y$ and $k$ is the no of distinct values which $w_i'$ can take. Here, $k$=2, because $w'$ may present or may not present in binary BoW. Typically $\alpha$=1

**Case 1 :** Let $\alpha$=1 with $n_1=100$. Then $P(w'|y=1)=\frac{0+1}{100+2*1}=\frac{1}{102}$
Now, $P(w'|y=1)\neq 0$. So, $P(y=1|text_q)\neq 0$
**Case 2 :** Let $\alpha$=1000 with $n_1=100$. Then $P(w'|y=1)=10000/20100\sim 1/2$
We are saying $P(w'|y=1)$ is same as $P(w'|y=0)$. Since we don't knnow the probability of that word in $y=1/0$, we are assuming it to be half. Same behaviour is applied even for words which are there in the corpus.

Let's say $P(x_i|y=1)=\frac{2}{50}$ without laplace smoothing.
With laplace smoothing,

|$\alpha$|with laplace smoothing, cond. prob. is|
|----|----|
|1|$\frac{3}{54}=5.555\%$|
|10|$\frac{12}{70}=17.14\%$|
|100|$\frac{102}{250}=40.8\%$|
|1000|$\frac{1000}{2050}=48.78\%$|

As $\alpha\ \uparrow$, we are moving the likelihood probabilities to the uniform distribution. If numerator & denominator is small, we have less confidence in the ratio, so we are giving higher $\alpha$.

Using $\alpha=1$ is called **add one smoothing**.

Why name smoothing? we are moving/smoothing the likelihood probs to the uniform distr.

###Log-probabilities for numerical stability

$0.2*0.1*0.2*0.1=0.0004$
Similarly consider having $100$ such numbers to find the probability, we'll have many zeros. It'll lead to **numerical underflow** in python as it only has 16 significant digits in float variable. Python will start doing rounding which causes errors.

Instead of using these probabilities, we can use the **log of these probabilities**.

$log(P(y=1|w_1,w_2,...w_d))=log(P(y=1)\prod_{i=1}^nP(x_i|y=1))$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$=log(P(y=1))+\sum_{i=1}^mlog(P(x_i|y=1))$ 
because $log(ab)=log(a)+log(b)$

As $x$ $\uparrow$, $log(x)$ $\uparrow$. So, we can compare the log of $P(y=1|x)$ and $P(y=0|x)$.

what is $2.2^{3.6}$? 17.089.. But how to find it. We need to use logs as
$log(2.2^{3.6})$
$=3.6*log(2.2)$
$=3.6*0.342422$
$=1.232721$

Take antilog for $0.2327$, we'll get $1709$. Now, add $.(dot)$ after the $1+1$ digit ending up with $17.09$

Another reason to use log is that, it converts **multiplication** into **addition**. And **exponentitaion** into **multiplication** making the computation much easier.

##Bias and Variance tradeoff

High bias $\longrightarrow$ Underfitting
High Variance $\longrightarrow$ Overfitting

We have only one hyper parameter : $\alpha$

Case 1 : $\alpha = 0$
Consider a case where we have very rare words (1 in 1000 ratio) and we still have probability for those. When we change the $D_{train}$ and that didn't have those rare word, the probability becomes zero when we fit for the test data and this is a big change. It is **overfitting or high variance**.

Case 2: $\alpha=$very large (like) $\alpha=10000$
$P(w_i|y=1)=2/1000$ becomes $P(w_i|y=1)=10002/21000 \sim 1/2$
Same is the effect for any other words, We can't draw a good difference as all the probabilities $P(x_i|y=0/1)$ are very near to 1/2. And we end up with decision which can be taken by $P(y=1/0)$, if $P(y=1) > P(y=0)$, we'll always get the answer as **+ve** class
It is **underfitting or high bias**.

How to find the right alpha $\alpha$?
We'll use it using **cross-validation** or **k-fold validation**.

##Feature importance and interpretability

For all words, find their conditional probabilities for each class and sort them. The word with the highest probability value is the most important feature/word.

$+ve$ class - Find words $w_i$ with highest value of $P(x_i|y=1)$
$-ve$ class - Find words $w_i$ with highest value of $P(x_i|y=0)$

***Interpretability:***

Given $x_q$ {$w_1,w_2\ ...\ w_d$} and we find $y_q=1$. We can conclude that $y_q=1$  because it has words $w_3, w_6, w_9$ with higher conditional probabilities.

##Imbalanced data

Consider $n_1$ positive class samples annd $n_2$ negative class samples. 

$P(y=1|w_1,w_2, ... w_d)=P(y=1)\prod_{i=1}^dP(x_i|y=1)$
$P(y=0|w_1,w_2, ... w_d)=P(y=0)\prod_{i=1}^dP(x_i|y=0)$
and $n_1\gg n_2$ such that the **prior** $P(y=1)=0.9$ and $P(y=0)=0.1$.

When we assume that **likelihood is same for both P(y=0/1)**, then the **priors will take advantage in the output**.

**Solutions:**
1) Upsampling or downsampling
2) Or simply drop $P(y=1) = P(y=0) = 1$
3) Modified NB. 

Consider 900 +ve class datapoints and 100 -ve class datapoints. and we have alpha as 10. For some word, the likelihood is 
$P(w_i|y=1)=18/900=0.02$ (without laplace smoothening) and $P(w_i|y=0)=2/100=0.02$

With laplace smoothening, it'll become $P(w_i|y=1)=28/920=0.03$ and $P(w_i|y=0)=12/120=0.1$. Here we can see that $\alpha$ gave more worth to minority -ve class.

We cann have a diff solution with $\alpha_1$ for +ve class and $\alpha_2$ for -ve class.

## Outliers

For text classification example, $w'$ not present during training. Laplace smoothing can take care of this. It is outlier in testing data.

What about during training phase? $w_8$ occurs **very very few** times.
**_Solution/hack_** : 
1. If a word occurs very less or less than say 10, just **remove that**.
2. Use **laplace smoothing**

##Missing values

For,
**Text data** (like amazon fine food reviews) : No case of missing data
**Categorical data** (like a climate type value missing) : Consider **NaN** as a category and proceed.
**Numerical data** : Take standard imputation methods or Gaussian NB.

##Handling Numerical features (Gaussian NB)

For real values features $\{f_1,f_2, ... f_d\}\ \epsilon\ R^d$
Let $x_{ij}$ be the real value of the feature $j$ in the $i^{th}$ data.
What is $P(x_{ij}|y=1)$?

We can plot the **PDF** for the feature $f_j$ in the class $y=1$ ($D'$ dataset with y=1). Get the probability from that curve. We **assume** that it is **Guassian Distribution** with N($\mu_j^1,\sigma_j^1$) for the +ve class and N($\mu_j^0,\sigma_j^0$) for the -ve class.

Here we can put any distribution like powerlaw, bernoulli based on our data.

##Multiclass classification

Same as before,
we can find 
$P(y=C_1|w_1,w_2..w_d)$
$P(y=C_2|w_1,w_2..w_d)$
....
$P(y=C_k|w_1,w_2..w_d)$

Take argmax and find the class.


##Similarity or Distance matrix
Can NB work given the distance or similarity matrix?

**NOPE** since NB doesn't use the distance based method while it uses the probability of features.

##Large dimensionality

NB is used extensively in BoW which itself has many dimensions. And we must **log probabilities** to handle the **number underflow**.

##Best and worst cases

1) Conditional independence assumption is done. If it is true, NB does well. While it is becoming false, NB starts deteriorating. Even if some features are dependent, NB works reasonably well.

2) For text classification problems (especially high dimensional data), NB works well and it'll act as benchmark.

3) Used extensively for categorical features.

4) NB is super interpretable and feature selection/importance. 

5) Timetaken for testing data is good for low-latency system.

6) we can easily overfit (if we don't do laplace smoothing)
