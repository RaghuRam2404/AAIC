# Functions

##Function declaration

Original ipynb : [https://drive.google.com/open?id=0BwNkduBnePt2RTR5VTRkYTBaemM](https://drive.google.com/open?id=0BwNkduBnePt2RTR5VTRkYTBaemM)

```
def function_name(arguments):
"""
Doc String
"""

Function statement(s)
Global variables (declared outside) can be accessed inside this function

return {var}
```

Executed as :
```
function_name(1,2)
```

Print the doc of the function
```
print(function_name.__doc__)
```

If no `return` statement, python returns `None` by default;

## Default arguments
```
def fname(arg1, arg2=True):
	"""
	First argument is mandatory. If the second one is not given, then value 'True' will be taken. 
	Have default value arguments at the end of mandatory arguments.
	"""
	
fname("test")
fname("test", False)
fname() # throws error
```

## Keyword arguments
```
def greet(**kwargs):
  if kwargs:
    print("Hello {}, your dob is {}".format(kwargs["name"], kwargs["dob"]))
greet(name="Raghu", dob="24 Apr,1993")

Output:
Hello Raghu, your dob is 24 Apr,1993
	
```

## Arbitrary arguments
```
def greet(*names):
  """
  No of arguments is not known
  """
  for name in names:
    print("Hello {0}".format(name))
greet("Raghu", "Vivek", "Bhavi")

Output:
Hello Raghu
Hello Vivek
Hello Bhavi

```

## Built-in Functions

| Function name | Description |
|---------|---------|
| `abs(num)` | returns the absolute value of a number|
|`all(var)`| returns `True` if all the elements in **var** are iterable. Eg: for **list** of elements, it'll return `True` if they are all non-zero or not-false. ![](./Functions/Screen Shot 2021-05-13 at 8.20.10 PM.png)![](./Functions/Screen Shot 2021-05-13 at 8.20.26 PM.png) |
| `dir(var)` | returns the list of valid attributes of the object **var**  ![](./Functions/Screen Shot 2021-05-13 at 8.22.44 PM.png) |
| `divmod(dividend, divisor)` | returns the **quotiend** and **remainder** as tuple. Eg `divmod(9,2)` returns `(4,1)` |
| `enumerate(var, start=0)` | adds index to the iterable and returns it. Note: start is a non-mandatory parameter. ![](./Functions/Screen Shot 2021-05-13 at 8.26.59 PM.png) |
| `filter(function,iterable)` | ![](./Functions/Screen Shot 2021-05-13 at 9.23.42 PM.png) |
| `isinstance(var, datatype)` | ![](./Functions/Screen Shot 2021-05-13 at 9.29.17 PM.png) |
| `map(function, iterable)` | ![](./Functions/Screen Shot 2021-05-13 at 9.31.53 PM.png) |
| <code>from functools import reduce<br>reduce(function, iterable)</code> | It applies the rolling computation for the sequential pairs of values in a list. ![](./Functions/Screen Shot 2021-05-13 at 9.44.06 PM.png) |
