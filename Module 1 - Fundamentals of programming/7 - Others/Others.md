# Other helpful libraries used

<script src="https://code.jquery.com/jquery-3.6.0.min.js" ></script>
<script src="../../toc.js" ></script>
<div id='toc'></div>

|Package|Function|
|---|----|
|matplotlib|colors.ListedColormap,pyplot.pcolormesh|
|BeautifulSoup||
|ScikitLearn|Normalize,CountVectorizer, TfidfVectorizer, [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier), [LocalOutlierFactor](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html), train\_test\_split|
|sklearn.sklearn.model_selection|[RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)|
|sklearn.metrics|cross\_val\_score, accuracy_score |
|sklearn.naivebayes|[MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html), [CategoricalNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html#sklearn.naive_bayes.CategoricalNB)|
|NLTK|PorterStemmer, stopwords, WordNetLemmatizer,FreqDist |
|gensim| Word2Vec |
|Re|compile,sub|
|mlxtend| plot\_decision\_regions |



##matplotlib

### ListedColormap

```
from matplotlib.colors import ListedColormap
scatter_cmap = ListedColormap(["#fcba03", "#1803fc"])
plt.scatter(X[:,0], X[:,1], c=Y, cmap= scatter_cmap)
plt.show()
```

### pyplot.pcolormesh

```
import matplotlib.pyplot as plt
h = 0.02
xmin, xmax = X[:,0].min()-1, X[:,0].max()+1
ymin, ymax = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.arange(xmin,xmax,h), np.arange(ymin,ymax,h))
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X,Y)
m_ = knn.predict(np.vstack((xx.ravel(), yy.ravel())).T)
m_ = m_.reshape(xx.shape)

mesh_cmap = ListedColormap(["#f5e7c4", "#dad7f7"])
plt.pcolormesh(xx, yy, m_, cmap=mesh_cmap)
```

## Beautiful Soup
Remove html tags from the text. Refer [3.1-Amazon Fine Food Reviews Analysis](https://colab.research.google.com/drive/1_GfKuT3_BtQlAxH7xmteQD0Sh9qqNOSu?authuser=1#scrollTo=R4eEj7-ZG6bV)

```
from bs4 import BeautifulSoup 
BeautifulSoup(sentence, "lxml").get_text() 
```

##ScikitLearn

###Normalize
Refer [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html)

```
from sklearn.preprocessing import normalize
normalize(X)
```

###CountVectorizer

Refer [3.1-Amazon Fine Food Reviews Analysis](https://colab.research.google.com/drive/1_GfKuT3_BtQlAxH7xmteQD0Sh9qqNOSu?authuser=1#scrollTo=STSHq40P-LQi), [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

```
from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer()
count_vec.fit(fdata['Text'])
fdata_bow = count_vec.transform(fdata['Text'])

all_words = count_vec.get_feature_names()
cx = fdata_bow[0]

for (index, count) in zip(cx.indices, cx.data):
  print("word {}".format("{"+all_words[index]+"}"), " with count : ", count)
```
###TfidfVectorizer

Refer [3.1-Amazon Fine Food Reviews Analysis](https://colab.research.google.com/drive/1_GfKuT3_BtQlAxH7xmteQD0Sh9qqNOSu?authuser=1#scrollTo=kNMGnU-fF4Z3), [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

```
from sklearn.feature_extraction.text import TfidfVectorizer

count_vec = TfidfVectorizer(ngram_range=(1,2))
count_vec.fit(fdata['Processed Text'])
fdata_tfidf = count_vec.transform(fdata['Processed Text'])
print(count_vec.idf_) #idf values
```

Refer [3.3 KNN](https://colab.research.google.com/drive/1iqJtquXlfDVC6YoBSTAjtlZH2uAOg04u?authuser=1#scrollTo=pnY19qYH98cu)

###KNeighborsClassifier

```
from sklearn.neighbors import KNeighborsClassifier
X_1, X_test, Y_1, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
model = KNeighborsClassifier(n_neighbors=k_val)
model.fit(X_1,Y_1)
y_pred = model.predict(X_test)
model.predict_proba(X_test)
```
Refer [3.3 KNN](https://colab.research.google.com/drive/1iqJtquXlfDVC6YoBSTAjtlZH2uAOg04u?authuser=1#scrollTo=azqnn2NI9bOO)

###cross\_val\_score and accuracy\_score

```
accuracy_scores = []

for k in k_range:
  scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, Y_train, scoring='accuracy', cv=10)
  accuracy_scores.append(np.mean(scores))

best_k_index = np.argmax(accuracy_scores)
best_k = k_range[best_k_index]
print(best_k)

#Generalization accuracy
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print("Generalization accuracy :",accuracy_score(Y_test, y_pred)*100.0,"%")
```

###LocalOutlierFactor

```
lof = LocalOutlierFactor(n_neighbors=3)
fp = lof.fit_predict(data[:,0:2])
filtered_data = data[np.where(fp == 1)]
```

###MultinomialNB

```
model = MultinomialNB(fit_prior=False)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
metrics.accuracy_score(Y_test, Y_pred) #0.97
Y_pred_proba = model.predict_proba(X_test)[:,1]
metrics.roc_auc_score(Y_test, Y_pred_proba) #0.97217
```

##NLTK
Refer [3.1-Amazon Fine Food Reviews Analysis](https://colab.research.google.com/drive/1_GfKuT3_BtQlAxH7xmteQD0Sh9qqNOSu?authuser=1#scrollTo=STSHq40P-LQi)

```
import nltk
```
###PorterStemmer

```
from nltk.stem import PorterStemmer
ps = PorterStemmer()
return ps.stem(word)
```
###Stop Words

```
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
```

###WordNetLemmatizer

```
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("bats")) #prints bat
```
###FreqDist

```
apw = nltk.FreqDist(all_positive_words)
print(apw.most_common(20))
```

##gensim

Refer [3.1-Amazon Fine Food Reviews Analysis](https://colab.research.google.com/drive/1_GfKuT3_BtQlAxH7xmteQD0Sh9qqNOSu?authuser=1#scrollTo=FCtdXsiIJE13)

```
from gensim.models import Word2Vec
w2v_model = Word2Vec(tqdm(list_of_sentences), min_count=5, size=100, workers=4)
w2v_model.wv.most_similar('flavor')
w2v_model.wv.similarity("flavor", "taste")
w2v_model.wv['taste']
```

##re

Refer [3.1-Amazon Fine Food Reviews Analysis](https://colab.research.google.com/drive/1_GfKuT3_BtQlAxH7xmteQD0Sh9qqNOSu?authuser=1#scrollTo=R4eEj7-ZG6bV), [https://pymotw.com/3/re/](https://pymotw.com/3/re/)

```
import re

re.sub(r'(http://\S+)|(https://\S+)', '', sentence)
#removes the http and https urls
```

##mlxtend

```
from mlxtend.plotting import plot_decision_regions
x = data[:,0:2]
y = data[:, 2]
model = KNeighborsClassifier(n_neighbors=k_val)
model.fit(x,y)
plot_decision_regions(x, y.astype(int), clf=model, legend=2, ax=ax)
```
