# Other helpful libraries used

|Package|Function|
|---|----|
|BeautifulSoup||
|ScikitLearn|CountVectorizer, TfidfVectorizer, KNeighborsClassifier, LocalOutlierFactor, train\_test\_split, cross\_val\_score, accuracy_score |
|NLTK|PorterStemmer, stopwords, WordNetLemmatizer,FreqDist |
|gensim| Word2Vec |
|Re|compile,sub|
|mlxtend| plot\_decision\_regions |

## Beautiful Soup
Remove html tags from the text. Refer [3.1-Amazon Fine Food Reviews Analysis](https://colab.research.google.com/drive/1_GfKuT3_BtQlAxH7xmteQD0Sh9qqNOSu?authuser=1#scrollTo=R4eEj7-ZG6bV)

```
from bs4 import BeautifulSoup 
BeautifulSoup(sentence, "lxml").get_text() 
```

##ScikitLearn
###CountVectorizer

Refer [3.1-Amazon Fine Food Reviews Analysis](https://colab.research.google.com/drive/1_GfKuT3_BtQlAxH7xmteQD0Sh9qqNOSu?authuser=1#scrollTo=STSHq40P-LQi)

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

Refer [3.1-Amazon Fine Food Reviews Analysis](https://colab.research.google.com/drive/1_GfKuT3_BtQlAxH7xmteQD0Sh9qqNOSu?authuser=1#scrollTo=kNMGnU-fF4Z3)

```
from sklearn.feature_extraction.text import TfidfVectorizer

count_vec = TfidfVectorizer(ngram_range=(1,2))
count_vec.fit(fdata['Processed Text'])
fdata_tfidf = count_vec.transform(fdata['Processed Text'])
print(count_vec.idf_) #idf values
```

Refer [3.3 KNN](https://colab.research.google.com/drive/1iqJtquXlfDVC6YoBSTAjtlZH2uAOg04u?authuser=1#scrollTo=pnY19qYH98cu)

```
from sklearn.neighbors import KNeighborsClassifier
X_1, X_test, Y_1, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
model = KNeighborsClassifier(n_neighbors=k_val)
model.fit(X_1,Y_1)
y_pred = model.predict(X_test)
acc = accuracy_score(Y_test, y_pred, normalize=True) * float(100)
acc_scores = cross_val_score(model, X_1, Y_1, cv=10, scoring="accuracy")
```
Refer [3.3 KNN](https://colab.research.google.com/drive/1iqJtquXlfDVC6YoBSTAjtlZH2uAOg04u?authuser=1#scrollTo=azqnn2NI9bOO)

```
lof = LocalOutlierFactor(n_neighbors=3)
fp = lof.fit_predict(data[:,0:2])
filtered_data = data[np.where(fp == 1)]
```

##NLTK
Refer [3.1-Amazon Fine Food Reviews Analysis](https://colab.research.google.com/drive/1_GfKuT3_BtQlAxH7xmteQD0Sh9qqNOSu?authuser=1#scrollTo=STSHq40P-LQi)

```
import nltk
```

```
from nltk.stem import PorterStemmer
ps = PorterStemmer()
return ps.stem(word)
```

```
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
```

```
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("bats")) #prints bat
```

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