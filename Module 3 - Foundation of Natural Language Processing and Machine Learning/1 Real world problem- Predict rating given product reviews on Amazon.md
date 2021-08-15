#Real world problem: Predict rating given product reviews on Amazon

<script src="https://code.jquery.com/jquery-3.6.0.min.js" ></script>
<script src="../toc.js" ></script>
<div id='toc'></div>

It is based on amazon food review data.

## Text Preprocessing

1. Removing stop-words.
2. Make all words lowercase.
3. Stemming.
4. Lemmitization.
5. Tokenization.

We can use fields 'Text' and 'Summary' for better analysis.

From Linear Algebra, we know that, if we have any input as vector, we can leverage all the mathematical learned in linear algebra.

How to convert simple text, words and sentences to numerical vectors?

Convert **Review** text to **d-dim** vector. Then we can draw **hyperplane** to do the classification.

![](./1 Real world problem- Predict rating given product reviews on Amazon/Screen Shot 2021-06-01 at 8.42.23 PM.png)

Text $\rightarrow$ $d$-dim vector, the **rules** are
> Suppose reviews $r_1$, $r_2$, $r_3$ has vectors $v_1$, $v_2$, $v_3$ correspondingly. If $r_1$ and $r_2$ are more similar **semantically** than $r_1$ and $r_3$, then the distance between $v_1$ and $v_2$ should be less than distance between $v_1$ and $v_3$. It means that similar points are closer.
>(i.e.) If sem($r_1$, $r_2$) > sem($r_1$, $r_3$), then dist($v_1$, $v_2$) < dist($v_1$, $v_3$)

Find {Text $\rightarrow$ $d$-dim vector}, such that similar text must be closer geometrically.

Each review is a **document**. And collection of documents is called **corpus**.

Such techniques are
1. Bag of Words (BoW)
2. tf-idf
3. Word2Vec

## Bag of Words (BoW)

Lets
$r_1$ : This pasta is very tasty and affordable
$r_2$ : This pasta is not tasty and is affordable
$r_3$ : This pasta is delicious and cheap
$r_4$ : Pasta is tasty and pasta tastes good


Steps :
1. Constructing a **dictionary** (set of all the words in your reviews). $d$-unique words across all reviews or document.
2. Each word is a different dimension, with $d$ dimensions for a document. For each review, we'll construct a vector with $d$ dimension with value as the **count of the corresponding words**. Let's say for $r_1$, the value is **1** for the index of the words "this", "pasta", "is", "very", "tasty", "and" & "affordable" and it'll be available in $v_1$. $v_1$ will be very sparse with most of the elements are *0*.

Now, we converted **text** to **vector**. If 2 documents are similar semantically, the resultant vectors must be closer.

![](./1 Real world problem- Predict rating given product reviews on Amazon/Screen Shot 2021-06-01 at 9.19.32 PM.png)

_Correction:_ **is** will occur only once. And the count will be put in the cell. So distance varies.

The length is very close. But their meanings are extremely opposite. This is where **BoW fails**, so we'll use **bi-gram or tri-gram**.

**Binary/Boolean BoW :** We'll put **1** if a number occurs else **0**. It'll return the **no. of differing words**.

## Text Preprocessing
###Stop Words

Also, the trivial words like **is**, **this**, **and** etc are not important. They are called **stop** words.

After removing those, the $v_i$ vectors will be small and more meaningful. Also we are throwing away some information. In english, **not** is a stop word. But removing it will cause problem.

```
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
```

###Stemming

**Tasty**, **tasting**, **tasteful** - instead of having them as 3 words in BoW, convert each of these words into their common form **tast** and replace those words with this. So many algorithms to do this like *Porter stemmer* (old one) and *snow ball stemmer* (latest one).

```
from nltk.stem.snowball import SnowballStemmer
snow_stemmer = SnowballStemmer(language='english')
print(snow_stemmer.stem("cared") #care
print(snow_stemmer.stem("university") #univers
print(snow_stemmer.stem("easily") #easili
```

### Lemmitization
Lemmatisation is the algorithmic process of determining the lemma of a word based on its intended meaning. Unlike stemming, lemmatization depends on correctly identifying the intended part of speech and meaning of a word in a sentence, as well as within the larger context surrounding that sentence, such as neighboring sentences or even an entire document. **Tasty**, **tasting**, **tasteful** will be converted to **taste**

Actually, lemmatization is preferred over Stemming because lemmatization does **morphological analysis** of the words.

```
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("bats")) #bat
print(lemmatizer.lemmatize("feet")) #foot
```

Read : [https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/](https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/) , [https://www.machinelearningplus.com/nlp/lemmatization-examples-python/](https://www.machinelearningplus.com/nlp/lemmatization-examples-python/)

### Tokenization
Breaking a sentence into words. We use "**space**" to separate sentence into words. But in case of word "New York", it is not possible as it is **language dependent** and **context dependent**. 

```
import nltk

word_data = "It originated from an idea"
nltk_tokens = nltk.word_tokenize(word_data)
print(nltk_tokens) #[It,originated,from,an,idea]
```

###Semantic meaning (not a preprocessing technique)

**Tasty** and **delicious** are synonyms.
We can make use of **Word2Vec** to achieve this synonyms.

**BoW** doesn't take the semantic meaning into the consideration.


### Spell check

1. Pasta is tasteeee
2. Pasta is tasty

[https://towardsdatascience.com/language-models-spellchecking-and-autocorrection-dd10f739443c](https://towardsdatascience.com/language-models-spellchecking-and-autocorrection-dd10f739443c) 
We need to take care of spell checks.

### Sample Code

[https://colab.research.google.com/drive/1xJYdhKwFAVuNR4ZLa7QScVb3EPoIg3HA?authuser=1#scrollTo=NA217LCLifB4](https://colab.research.google.com/drive/1xJYdhKwFAVuNR4ZLa7QScVb3EPoIg3HA?authuser=1#scrollTo=NA217LCLifB4)

```
import unicodedata
import re
from sklearn.feature_extraction import text
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess(txt):

  """Preprocesses the given document by applying the following functionalities
  lower: lowers all the characters for uniformity
  expansion: expands words like i'll to i will for better text classification
  remove special characters: using regex, removes all the punctuations etc
  remove space: removes trailing spaces and extra spaces between words
  remove accented characters: change accented characters to its normal equivalent
  remove stop words: removes the stop words in the sentence
  lemmatization: changes the words to their base form"""

  #to lower case and other custom removal
  txt = txt.lower().replace('numbertnumber','').replace('number', '').replace('date','').replace('url','').strip()

  #expansion: expands words like i'll to i will for better text classification
  for key in contractions:
    search_for = key.replace("'",' ')
    txt.replace(search_for, contractions.get(key))  

  #remove special characters: using regex, removes all the punctuations etc
  txt = re.sub("[^A-Z a-z 0-9-]+",'',txt)

  #remove space: removes trailing spaces and extra spaces between words
  txt = ' '.join(txt.split())

  #remove accented characters: change accented characters to its normal equivalent
  txt = unicodedata.normalize('NFKD', str(txt)).encode('ascii', 'ignore').decode('utf-8', 'ignore')

  #remove stop words: removes the stop words in the sentence
  ss = stopwords.words("english")
  if "no" in ss:
    ss.remove("no")
  if "not" in ss:
    ss.remove("not")
  txt = ' '.join([wrd if wrd not in ss else "" for wrd in txt.split()])

  #lemmatization: changes the words to their base form
  wnl = WordNetLemmatizer()
  sbs = SnowballStemmer(language="english")
  txt = ' '.join([wnl.lemmatize(ss) for ss in txt.split()])

  return txt  
```


## uni-gram, bi-gram, n-grams.

$r_1$ : This pasta is very tasty and affordable
$r_2$ : This pasta is not tasty and is affordable

After removing stop words (this, is, not, very, and), $v_1$ and $v_2$ are exactly the same. Then we'll consider the both are similar but in reality, they are not.

We can fix it by 'bi-gram' or 'tri-gram' as unigram BoW discard the sequence of information. In **unigram**, each word will be considered a dimension. In **bi-gram**, pair of words will form a dimension. So 'This pasta', 'pasta is', 'is very', 'very tasty' and so on will form dimensions. Inn **tri-gram**, take 3 consecutive words as one dimension and it follows for **n-gram**. Other than, unigram, others will retain some of the sequential information.

If **there are repeated words**, the dimensions in `#n-gram` $\geq$ `#tri-gram` $\geq$ `#bi-gram` $\geq$ `#uni-gram`.

Eg :
**horse is a horse, of course, of course. accept  it**
Here there are repeated words
unigrams --> horse, of, course, a, is, it, accept --> #7
bigrams --> horse is, is a , a horse, horse of, of course, course of, course accept, accept it --> #8
trigrams --> horse is a, is a horse, a horse of, horse of course, of course of, course of course, of course accept, course accept it --> #8

As **n-gram increases**, the **dimensionality d increases**.

Ref : [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

```
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['The pasta taste good', 'This pasta doesnot taste good']
cv = CountVectorizer(corpus, stop_words=['this', 'is', 'the'])
cv.fit(corpus)
print(cv.get_feature_names())
print(cv.transform(corpus).toarray())
print(cv.vocabulary_)
print(cv.token_pattern)

['good', 'not', 'pasta', 'tastes']
[[1 0 1 1]
 [1 1 1 1]]
{'pasta': 2, 'tastes': 3, 'good': 0, 'not': 1}
(?u)\b\w\w+\b
```

```
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['This pasta tastes good', 'This pasta tastes not good']
cv = CountVectorizer(corpus, stop_words=['this'], ngram_range=(2,2))
cv.fit(corpus)
print(cv.get_feature_names())
print(cv.transform(corpus).toarray())
print(cv.vocabulary_)
print(cv.token_pattern)

['not good', 'pasta tastes', 'tastes good', 'tastes not']
[[0 1 1 0]
 [1 1 0 1]]
{'pasta tastes': 1, 'tastes good': 2, 'tastes not': 3, 'not good': 0}
(?u)\b\w\w+\b
```

## tf-idf (term frequency- inverse document frequency) BoW

It belongs to **information retrieval** sub area of NLP.

This method doesn't take the semantic meaning into the consideration.

Let's assume we have $N$ documents/reviews.

**Term frequency** of word $W_i$ occuring in the document/review $r_j$ is
$TF=\frac{\#\ of\ times\ W_i\ occurs\ in\ r_j}{total\ \#\ of\ words\ in\ r_j}$

$D_c$ is set of all the below reviews. It is **corpus**
$r_1$ is $W_1$, $W_2$, $W_3$, $W_2$, $W_5$
$r_2$ is $W_1$, $W_2$, $W_4$, $W_3$, $W_5$, $W_6$
...
$r_N$

$TF(W_2,r_1)=\frac{2}{5}$

$0\ \leq\ TF\ \leq\ 1$, so it is like probability of finding a word in the document.

**IDF** : If a word occurs in many documents, then IDF is low. If a word is a rare or low frequency, then IDF is high.

$IDF(W_i, D_c)$ = $log(\frac{N}{n_i})$ where $n_i$ is the no of documents which has the word $W_i$ and $N$ is the no of total documents.

$n_i\ \leq\ N\ \Longrightarrow\ \frac{N}{n_i}\geq1\ \Longrightarrow\ IDF=log(\frac{N}{n_i})\geq0$
As $n_i$ increases, $\frac{N}{n_i}$ and $log(\frac{N}{n_i})$ both decreases.

**It means that if a word occurs more frequently in the corpus, then the $IDF$ value is smaller.**
(i.e.) if $n_i\uparrow$, $IDF\downarrow$. Also if $n_i\downarrow,IDF\uparrow$

Usually in **BoW**, we'll fill the vector with the count of words.
But now, we'll fill **index** corresponding to the word $W_i$ with **$TF(W_i, r_j)*IDF(W_i,D_c)$** in the document $r_j$. It gives more weightwage to the most frequent words in a document and less weightage to the common words in the corpus.

**More importance to the rarer word in the corpus and to the frequent word in a document.**

It still doesn't take into account the semmantic meaning of the word.

Ref : [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

```
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ['This pasta tastes good', 'This pasta tastes not good']
tfidf = TfidfVectorizer(corpus, ngram_range=(2,2), stop_words=['this'])
tfidf.fit(corpus)
print(tfidf.get_feature_names())
print(tfidf.idf_)
print("------------"*4)
print(tfidf.transform(corpus).toarray())
print(tfidf.vocabulary_)

['not good', 'pasta tastes', 'tastes good', 'tastes not']
[1.40546511 1.         1.40546511 1.40546511]
------------------------------------------------
[[0.         0.57973867 0.81480247 0.        ]
 [0.6316672  0.44943642 0.         0.6316672 ]]
{'pasta tastes': 1, 'tastes good': 2, 'tastes not': 3, 'not good': 0}
```

###Why use log in IDF?

As per Zipf's law, the common words in english will occur much more and rarer words will be less. It follows **power-law** distribution.
![](./1 Real world problem- Predict rating given product reviews on Amazon/Screen Shot 2021-06-07 at 11.06.23 AM.png)
![](./1 Real world problem- Predict rating given product reviews on Amazon/Screen Shot 2021-06-07 at 11.07.53 AM.png)
Instead of giving weightage 1000 to the word **civilization**, we are just giving $6.9$ .

**_Both BoW and tf-idf convert a full text/document into a sparse vector._**

## Word2Vec

Takes the **semantics into the consideration**. It is the state of the art technique. It learns relationships automatically from raw text.

Given a **word**, it'll be converted to **d-dimensional dense vector** (typically d will be 100,200,300). Note that, here word is converted to a vector not a sentence or document.

Take 3 words "tasty", "delicious", "baseball". As Word2Vec knows English, it'll return 3 vectors $v_1,v_2$ and $v_3$ respectively such that the distance between first 2 are closer and $v_3$ is far.

1. So, given any 2 semantically similar words, Word2Vec converts them to 2 vectors with minimal distance.
2. Relationships are satisfied. Take words man, woman, king and queen, their vectors $v_{man}$, $v_{woman}$, $v_{king}$ and $v_{queen}$, then ($v_{man}$-$v_{woman}$) vector is parallel to ($v_{king}$-$v_{queen}$) vector. Same applicable for country-capital, verb tense, gender and so on. 

![](./1 Real world problem- Predict rating given product reviews on Amazon/Screen Shot 2021-06-07 at 11.45.52 AM.png)


![](./1 Real world problem- Predict rating given product reviews on Amazon/Screen Shot 2021-06-07 at 11.50.18 AM.png)

As corpus size increases, the d-dimensions will also increase with information rich vectors.

Intuitively, it works like if the neighborhood of the word $W_i$ is similar to the word $W_j$, then $v_i$ is similar to $v_j$
(i.e) $N(W_i)\approx N(W_j)$  $\Longrightarrow$ $v_i\approx v_j$

Ref : [https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial](https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial), [http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.YQ4eEVMzZQJ](http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.YQ4eEVMzZQJ)

```
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
word = 'McLaren'
if word in w2v.vocab.keys():
  print(w2v.similar_by_word(word, topn=5))
  w2v_vec = w2v.get_vector(word)
  print(np.array(w2v_vec).shape)
  
[('McLaren_Mercedes', 0.7571395635604858), 
('Brawn', 0.7284218668937683), 
('Mclaren', 0.6954721808433533), 
('BAR_Honda', 0.688281774520874), 
('Ferrari', 0.6736721992492676)]

(300,)
```

```
#Own w2v model
lines = []
for line in data['essay'].values:
  lines.append(line.split())
w2v = Word2Vec(sentences=lines,min_count=5 ,size=50, workers=4)

print(w2v.wv.most_similar('great', topn=2))
#[('excellent', 0.7378597855567932), ('wonderful', 0.7312985062599182)

print(w2v.wv.vocab) #full vocab

print(w2v.wv['great'])
#prints the vector

w2v.wv.similarity('great', 'worst') #0.07923318
w2v.wv.similarity('great', 'excellent') #0.73785985

```

### Using sentences, how to convert using word2vec?
Using Avg-Word2Vec, tf-idf weighted Word2Vec weighing strategies.

####Avg-Word2Vec
$r_1$ : $W_1$ $W_2$ $W_1$ $W_3$ $W_4$ $W_5$ 

Then $r_1\rightarrow v_1$ is $\frac{W2V(W_1)+W2V(W_2)+W2V(W_1)...W2V(W_5)}{n_1}$

This is averaging all the vectors. It is not perfect. It is well enough.

```
line = "The pasta tastes good"
count = 0
vec = np.zeros(w2v.wv.vector_size)
for word in line.split():
  if word in w2v.wv.vocab:
    vec += w2v.wv[word]
    print(word)
    count += 1
if count > 0:
  vec /= count
print(vec) #prints 1xn vector if the dimension is n
```


####tf-idf weighted Word2Vec

Compute the **tf_idf** for $r_1$, then multiply each corresponding value with the $W2V(W_i)$

![](./1 Real world problem- Predict rating given product reviews on Amazon/Screen Shot 2021-06-07 at 12.12.55 PM.png)

$tfidf_w2v(r_1)=\frac{\sum_{i:words}t_i*w2v(W_i)}{\sum_{i:words}t_i}$

```
line = "The pasta tastes good"
tfidf = TfidfVectorizer()
tfidf.fit(lines)

idf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
w2v_vocab = w2v.wv.vocab
tf_c = 0
vec = np.zeros(w2v.wv.vector_size)

words = line.split()
for word in words:
  if word in w2v_vocab and word in idf_dict:
    tf = words.count(word)/len(words)
    tfidf_val = tf * idf_dict[word]
    vec += tfidf_val * w2v.wv[word]
    tf_c += tfidf_val

if tf_c > 0:
  vec /= tf_c

print(vec)
```
