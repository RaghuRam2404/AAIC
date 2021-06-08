# Other helpful libraries used

|Package|Function|Help|
|---|----|---|
|ScikitLearn|CountVectorizer|Using BoW|
|Re|compile,sub|[https://pymotw.com/3/re/](https://pymotw.com/3/re/)|


#ScikitLearn
##CountVectorizer
```
from sklearn.feature_extraction.text import CountVectorizer
```

```
count_vec = CountVectorizer()
count_vec.fit(fdata['Text'])
print("some feature names ", count_vec.get_feature_names()[200:210])

fdata_bow = count_vec.transform(fdata['Text'])
print("Type of fdata_bow  : ", type(fdata_bow))
print("BoW matrix's shape : ", fdata_bow.get_shape())
print("No of unique words : ", fdata_bow.shape[1])
```
```
some feature names  ['30th', '31', '32', '320', '32oz', '33', '330mg', '336', '34', '349']
Type of fdata_bow  :  <class 'scipy.sparse.csr.csr_matrix'>
BoW matrix's shape :  (4986, 13510)
No of unique words :  13510
```
```
all_words = count_vec.get_feature_names()
print(fdata.iloc[0]['Text'])
cx = fdata_bow[0]

for (index, count) in zip(cx.indices, cx.data):
  print("word {}".format("{"+all_words[index]+"}"), " with count : ", count)
```
```
I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most.
word {all}  with count :  1
word {and}  with count :  3
----and so on---
```


#NLTK

#re
