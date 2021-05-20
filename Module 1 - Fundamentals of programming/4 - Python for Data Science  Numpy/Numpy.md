# Numpy

1. Extension package to python for multidimensional arrays
2. Closer to hardware efficiency
3. Designed for scientific computation
4. Known as array oriented computing

`import numpy as np`

Array Create

- `np.array([1,2,3])`
- `np.arange(10)` - create numbers from 0 to 9
- `np.arange(1,11)` - create numbers from 1 to 10
- `np.arange(0,11,2)` - create numbers from 0 to 10 with stepsize as 2. so **[0, 2, 4, 6, 8, 10]**
- `np.arange(10, dtype='float64')` - create floating numbers
- `np.linspace(0,1,11)` - create linearly spaced array
- `np.ones(4)` or `np.ones((2,3))` - Matrix with ones. Send the matrix size as integer for 1 dimension and as tuple for n-dimension
- `np.zeros((2,3))` - Matrix with zeros. Send the matrix size as tuple
- `np.eye(3)` or `np.eye(3,2)` - Identity matrix 3x3 or matrix with ones only in the first 2 rows
- `a = np.diag([1,2,3,4])` - create diagonal matrix
- `np.diag(a)` - extract the diagonal matrix elts in a list
- `np.random.rand(4)` - random numbers of uniform distribution
- `np.random.randn(4)` - random number of standard deviation
- `np.random.randint(0, 20, 15)` - random integers [start index, end index, total number of elements]
- `np.zeros_like(a)` - Return an array of zeros with the same shape and type as a given array.
- `np.cumsum(a)` - cummulative sum
- `np.mean(a)` - find the mean (average) of elements
- `np.median(a)` - find the median of elements
- `np.percentile(a, reqd_percentiles)` - find the percentile values passed in the `reqd_percentiles` array or scalar


Array's attributes

- `.ndim` - dimension
- `.shape` - shape as matrix
- `.dtype` - gives the data type
- `.T` - will give the transpose

Array's functions

- `len(nparray)` - returns the no of element in the first dimension
- `np.sum(a)` - returns the sum of all elements in an array
- `np.sum(a, axis=0/1)` - returns the array of sum of elements in each axis
- `np.shares_memory(a,b)` - Boolean : True if both **a** and **b** shares same RAM memory
- `np.sin(a)` - Elementwise apply **sin** function
- `np.log(a)` - Elementwise apply **log** function
- `np.exp(a)` - Elementwise apply **exp** function
- `np.min(a)` , `np.max(a)` - Find min & max of the array
- `np.argmin(a)` , `np.argmax(a)` - Find the argument of min & max element of the array
- `np.all(a)` - Boolean : true if all the values are true
- `np.any(a)` - Boolean : False if all the values are False
- `np.mean(x)` or `np.mean(x,axis=0/1)` - Find mean for 1D or 2D array
- `np.median(x)` or `np.median(x, axis=0/1)` - Find median for 1D or 2D array
- `np.std(x)` or `np.std(x, axis=0/1)` - Find standard deviation for 1D orr 2D array
- `array.ravel()` - Flatten the matrix by iterating each dimensions
- `array.reshape((1,2))` - Send a **tuple** as shape to reshape. The total elements in the matrix should match the product of nos. in the tuple. ***It may create a copy or create a view***. So be careful
- `array.resize((1,2))` - Send a **tuple** as shape to resize. The total elements in the matrix may not match the product of nos. in the tuple. So if needed, 0's will be added.
- `np.sort(x)` or `np.sort(x, axis=0/1)` - sort the numpy array **x**
- `np.argsort(x)` or `np.argsort(x, axis=0/1)` - return the **sorted indices**



## Creating Arrays
```
b = np.array([[0,1,2], [3,4,5]])
print(b.ndim)
print(b.shape)
print(b.dtype)

2
(2,3)
int64
```

```
a = np.linspace(0,1,11) #start, end, no of points
a

array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
```

```
print(np.ones((3,3)))

[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
```

```
print(np.zeros((3,3)))

[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
```

```
a = np.eye(3)
print(a)

[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

```
a = np.diag([1,2,3,4]) #diagonal matrix
print(a)

[[1 0 0 0]
 [0 2 0 0]
 [0 0 3 0]
 [0 0 0 4]]

```

## Indexing and Slicing
```
a = np.arange(10)
print(a[5])

5
```

```
b = np.diag([1,2,3,4,5])
print(b[2,2])
print(b[2][2])

3
3
```

```
b[2,1] = 10
print(b)

[[ 1  0  0  0  0]
 [ 0  2  0  0  0]
 [ 0 10  3  0  0]
 [ 0  0  0  4  0]
 [ 0  0  0  0  5]]
```

```
a = np.arange(11)
print(a[2:10])
print(a[:10])
print(a[0:10:3]) #with step value

[2 3 4 5 6 7 8 9]
[0 1 2 3 4 5 6 7 8 9]
[0 3 6 9]
```

```
a[5:] = 10
print(a)

[ 0  1  2  3  4 10 10 10 10 10 10]
```

```
b = np.arange(6)
a[5:] = b[:]
print(a)
a[5:] = b[::-1] #in reverse order
print(a)

[0 1 2 3 4 0 1 2 3 4 5]
[0 1 2 3 4 5 4 3 2 1 0]
```

## Copies and Views

```
a = np.arange(10)
b = a[::2] #it is a view of a
b[0] = 10
print(b)
print(a)
print(np.shares_memory(a,b))

[10  2  4  6  8]
[10  1  2  3  4  5  6  7  8  9]
True
```

```
a = np.arange(10)
c = a[::2].copy()
print(c)
c[0] = 10
print(a)
print(c)

[0 2 4 6 8]
[0 1 2 3 4 5 6 7 8 9]
[10  2  4  6  8]
```

##Fancy Indexing
```
a = np.random.randint(0, 20, 15)
print(a)
mask = (a%2 == 0)
print(mask)

[ 6 11 11 13  9  2  5  9  6 11  1 13 15 14 18]
[ True False False False False  True False False  True False False False
 False  True  True]
 
```

```
from_a = a[mask] #creates a copy but not view
print(from_a)
from_a[0] = 11
print(from_a)
print(a)

[ 6  2  6 14 18]
[11  2  6 14 18]
[ 6 11 11 13  9  2  5  9  6 11  1 13 15 14 18]
```

```
a[mask] = -1
print(a)

[-1 11 11 13  9 -1  5  9 -1 11  1 13 15 -1 -1]
```

Array's elements can be selected by sending the **list of indices** as index

```
a = np.arange(0,100,10)
new_a = a[[2,3,4,3,2]] #choose the 2nd, 3rd, 4th, 3rd, 2nd and create a copy
print(new_a)

[20 30 40 30 20]
```

```
a = np.arange(0,100,10)
a[[0, 1, 2, 3]] = [100, 200, 300, 400]
print(a)

[100 200 300 400  40  50  60  70  80  90]
```

## Numpy Operations

```
a = np.array([1,2,3,4])
print(a+1)

[2 3 4 5]
```

```
a = np.array([1,2,3,4])
b = np.ones(4) + 1
a-b #elementwise subtraction
a*b #elementwise multiplication

[-1.,  0.,  1.,  2.]
[2., 4., 6., 8.]
```

```
c = np.diag([1,2,3,4])
d = np.ones((4,4))+2
c[0,3] = 4
d[2:3] = d[2:3]+1
d[0,3] = 5

print(c)
print(d)
print("----------RESULT-----------")
print(c*d) #elementwise multiplication provided both matrices have same shape
print(c.dot(d)) #matrix multiplication (i.e. for each elt in the result matrix, we'll do the dot product of each row and column)

[[1 0 0 4]
 [0 2 0 0]
 [0 0 3 0]
 [0 0 0 4]]
[[3. 3. 3. 5.]
 [3. 3. 3. 3.]
 [4. 4. 4. 4.]
 [3. 3. 3. 3.]]
----------RESULT-----------
[[ 3.  0.  0. 20.]
 [ 0.  6.  0.  0.]
 [ 0.  0. 12.  0.]
 [ 0.  0.  0. 12.]]
[[15. 15. 15. 17.]
 [ 6.  6.  6.  6.]
 [12. 12. 12. 12.]
 [12. 12. 12. 12.]]

```

```
c > d #elementwise comparison

[[ True,  True,  True,  True],
       [ True,  True,  True,  True],
       [ True,  True,  True,  True],
       [ True,  True,  True, False]]
```

```
a = np.array([1,2,3,4])
b = np.array([1,5,4,3])
c = np.array([1,2,3,4])

print(np.array_equal(a,b)) 
print(np.array_equal(a,c)) #all elts are same elementwise

False
True
```

```
a = np.array([1,1,0,0], dtype="bool")
b = np.array([1,0,1,0], dtype="bool")
print(np.logical_or(a,b))
print(np.logical_and(a,b))

[ True  True  True False]
[ True False False False]
```

## Numpy Functions
```
x = np.array([[1,2,3,4],[5,6,7,8]])
print(np.sum(x))
print(np.sum(x, axis=0)) #columnwise addition, because column is the innermost
print(np.sum(x, axis=1)) #rowwise addition

36
[ 6  8 10 12]
[10 26]
```

```
print(np.all([True, True, False]))
print(np.any([True, False, False]))

False
True
```

```
a = np.zeros((50,50))
print(np.any(a>0))

False
```

```
a = np.array([1,2,3,2])
b = np.array([2,2,3,2])
c = np.array([6,4,4,5])
((a <= b) & (b <= c)).all()

True
```

## Broadcasting

![](./Numpy/broadcasting.png)

```
a = np.tile(np.array([0,10,20,30]), (3,1)) #in the new matrix, have entire row 3 times and entire column 1 time
print(a)
b = np.array([0,1,2,3])
print(b)

[[ 0 10 20 30]
 [ 0 10 20 30]
 [ 0 10 20 30]]
[0 1 2 3]
```

```
a+b #array b is broadcasted to new rows (same no of rows as in a)

array([[ 0, 11, 22, 33],
       [ 0, 11, 22, 33],
       [ 0, 11, 22, 33]])
```

```
a.T + np.array([0,1,2])

array([[ 0,  1,  2],
       [10, 11, 12],
       [20, 21, 22],
       [30, 31, 32]])
```

```
a = np.arange(0, 40, 10)
print(a.shape)
print(a, '\n--------------')
a = a[:, np.newaxis]
print(a.shape)
print(a)

b = np.array([0, 1, 2])
print(b)

print(a+b)

(4,)
[ 0 10 20 30] 
--------------
(4, 1)
[[ 0]
 [10]
 [20]
 [30]]
[0 1 2]
[[ 0  1  2]
 [10 11 12]
 [20 21 22]
 [30 31 32]]
```

## Flattening

```
a = np.array([[1,2,3,4],[5,6,7,8]])
print(a)
print(a.ravel()) #Flattening
print(a.T.ravel())

[[1 2 3 4]
 [5 6 7 8]]
[1 2 3 4 5 6 7 8]
[1 5 2 6 3 7 4 8]

```

## Reshape
```
a.reshape((2,2,2))

array([[[1, 2],
        [3, 4]],

       [[5, 6],
        [7, 8]]])
```

## Resizing

```
a = np.arange(4)
a.resize((8,))
print(a)

[0 1 2 3 0 0 0 0]
```

```
a.resize((3,))
print(a)

[0 1 2]
```

```
a = np.arange(4)
b = a
a.resize((8,)) #Not allowed as a is referrenced by b
print(a)

### ERROR
```

## Sorting

```
a = np.array([[5,4,6], [2,3,2]])
b = np.sort(a, axis=1)
print(b)

[[4 5 6]
 [2 2 3]]
```

```
b = np.sort(a, axis=0)
print(b)

[[2 3 2]
 [5 4 6]]
```

```
a = np.array([4,3,6,8,1,0])
c = np.argsort(a) #returns the sorted indices
print(c)
print(a[c])

[5 4 1 0 2 3]
array([0, 1, 3, 4, 6, 8])
```