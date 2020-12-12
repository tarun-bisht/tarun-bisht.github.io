---
layout: writing
title:  crash course On numpy library python for everyday data science
date:   2020-04-13 22:42:50 +0530
category: Crash Course
tags: beginner python
---
NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. This post acts as a cheat-sheet for using the NumPy library in python. It contains some important functions and submodules of the NumPy library which are used day to day in data science and machine learning.
<!-- more -->

Merits of numpy over python list: --

1. Occupy less memory compare to list
2. Fast compared to the list
3. Convenient to use.

### Installing NumPy  

An easy way to install numpy is via pip. Create a virtual environment and activate it or activate an old virtual environment and run command to install


{% highlight bash linenos %}
pip install numpy
{% endhighlight %}

### Importing library


{% highlight python linenos %}
import numpy
{% endhighlight %}

numpy can be imported by an alias name np or any which you like


{% highlight python linenos %}
import numpy as np
{% endhighlight %}

### Creating NumPy arrays

#### array

syntax: *np.array(list_of_elements)*

Creates a numpy array from the list passed as parameter


{% highlight python linenos %}
var=np.array([[2,3],[3,2]])
print(var)
{% endhighlight %}

    [[2 3]
     [3 2]]
    

#### arange

syntax: 
* *np.arange(number_of_iter)*
* *np.arange(start,stop,step,dtype)*

Same as range() which creates an array of numbers between the range specified


{% highlight python linenos %}
var=np.arange(100)
print(var)
{% endhighlight %}

    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
     48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
     72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
     96 97 98 99]
    

#### diag

syntax: *np.diag(list_of_values)*

Creates a diagonal array of values provided in list send as parameter.


{% highlight python linenos %}
var=np.diag(range(4))
print(var)
{% endhighlight %}

    [[0 0 0 0]
     [0 1 0 0]
     [0 0 2 0]
     [0 0 0 3]]
    

#### linspace

syntax: *np.linspace(value1,value2,number_of_elements)*
It will return a numpy array which contains numbers of elements passed in number_of_elements parameter whose values is between *value1* and *value2* which are at equal interval. 

Say let *value1 = 1* and *value2 = 3* and we need 5 elements then we get 

[1, 1.5, 2, 2.5, 3]


{% highlight python linenos %}
var=np.linspace(1,2,5)
print(var)
{% endhighlight %}

    [1.   1.25 1.5  1.75 2.  ]
    

#### random.randn

syntax: *np.random.randn(dimensions)*

Generates an array with random numbers


{% highlight python linenos %}
var=np.random.randn(2,2,3)
print(var)
{% endhighlight %}

    [[[ 0.62417698 -0.53171972 -0.25723222]
      [ 1.36357605 -0.43352711  0.19280647]]
    
     [[ 0.36747904  1.20299115  0.95669774]
      [ 0.13572886  0.4454223  -1.07577104]]]
    

#### random.normal

syntax: *np.random.normal(size=(dimensions))*

Generates an array with random numbers from a normal (Gaussian) distribution


{% highlight python linenos %}
var=np.random.normal(size=(2,2))
print(var)
{% endhighlight %}

    [[ 0.09112449  0.9688921 ]
     [ 1.09627332 -1.0526527 ]]
    

#### zeros

syntax: *np.zeros((dimension))*
    
Generates a zero or null array with all elements = 0 of dimension specified.


{% highlight python linenos %}
var=np.zeros((2,2))
print(var)
{% endhighlight %}

    [[0. 0.]
     [0. 0.]]
    

#### ones

syntax: *np.ones((dimension))*
    
Generates a array with all elements = 1 of dimension specified.


{% highlight python linenos %}
var=np.ones((2,2))
print(var)
{% endhighlight %}

    [[1. 1.]
     [1. 1.]]
    

#### identity

syntax: *np.identity(dimension)*
    
Generates an identity matrix with dimension specified


{% highlight python linenos %}
var=np.identity(2)
print(var)
{% endhighlight %}

    [[1. 0.]
     [0. 1.]]
    

### Important properties of NumPy array

For reference let



{% highlight python linenos %}
var= np.array([[2,3],[4,5]])
print(var)
{% endhighlight %}

    [[2 3]
     [4 5]]
    

#### ndim

gives out the dimension of the array


{% highlight python linenos %}
var.ndim
{% endhighlight %}

    2

#### itemsize

Gives out number of elements in array


{% highlight python linenos %}
var.itemsize
{% endhighlight %}


    4


#### dtype

gives out data type of elements


{% highlight python linenos %}
var.dtype
{% endhighlight %}


    dtype('int32')


#### size

gives out the size of array ie. Number of elements in the array


{% highlight python linenos %}
var.size
{% endhighlight %}




    4



#### shape

gives out the dimension of array in the form of tuple


{% highlight python linenos %}
var.shape
{% endhighlight %}




    (2, 2)



#### reshape

reshape array dimensions to the desired one if the given shape is compatible. Make sure the product of dimension before and after is the same. ie.. If we have an array of dimension (2,2) in this case product of dimension in 2x2=4 so it can be reshaped into (1,4) since product of dimension is still 2X1=4


{% highlight python linenos %}
a=var.reshape((1,4))
print(a)
{% endhighlight %}

    [[2 3 4 5]]
    


{% highlight python linenos %}
a=var.reshape((1,2))
print(a)
{% endhighlight %}


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-20-62745c373b92> in <module>
    ----> 1 a=var.reshape((1,2))
          2 print(a)
    

    ValueError: cannot reshape array of size 4 into shape (1,2)


### Maths Functions in NumPy

* #### sqrt

Find square of each element present in numpy array

* #### sin

finds sine value of all the elements of the array

* #### cos 

finds cosine value of all the elements of the array

* #### tan
finds the tangent value of all the elements of the array

* #### log

finds log value of all the elements of the array

* #### log10

finds log base 10 value of all the elements of the array

* #### exponent

finds exponents (e raise to the power) value of all the elements of the array

* #### standard deviation

return standard deviation of elements of the array

* #### mean

return mean of elements of the array

* #### variance

return variance of elements of the array

* #### maximum

return max value in the array from axis specified

* #### sum

return sum of all elements of numpy array from the axis specified


{% highlight python linenos %}
print(np.sqrt(var))
{% endhighlight %}

    [[1.41421356 1.73205081]
     [2.         2.23606798]]
    


{% highlight python linenos %}
print(np.sin(var))
{% endhighlight %}

    [[ 0.90929743  0.14112001]
     [-0.7568025  -0.95892427]]
    


{% highlight python linenos %}
print(np.cos(var))
{% endhighlight %}

    [[-0.41614684 -0.9899925 ]
     [-0.65364362  0.28366219]]
    


{% highlight python linenos %}
print(np.tan(var))
{% endhighlight %}

    [[-2.18503986 -0.14254654]
     [ 1.15782128 -3.38051501]]
    


{% highlight python linenos %}
print(np.log(var))
{% endhighlight %}

    [[0.69314718 1.09861229]
     [1.38629436 1.60943791]]
    


{% highlight python linenos %}
print(np.log10(var))
{% endhighlight %}

    [[0.30103    0.47712125]
     [0.60205999 0.69897   ]]
    


{% highlight python linenos %}
print(np.exp(var))
{% endhighlight %}

    [[  7.3890561   20.08553692]
     [ 54.59815003 148.4131591 ]]
    


{% highlight python linenos %}
print(np.std(var))
{% endhighlight %}

    1.118033988749895
    


{% highlight python linenos %}
print(np.mean(var))
{% endhighlight %}

    3.5
    


{% highlight python linenos %}
print(np.var(var))
{% endhighlight %}

    1.25
    


{% highlight python linenos %}
print(np.max(var,axis=None))
{% endhighlight %}

    5
    


{% highlight python linenos %}
print(np.sum(var,axis=None))
{% endhighlight %}

    14
    

### NumPy Operations Functions

#### vstack

syntax: *np.vstack((arrays_to_stack_verical))*

stacks vertically all the arrays send in parameter


{% highlight python linenos %}
a=np.zeros(shape=var.shape)
vertical=np.vstack((var,a))
print(vertical)
{% endhighlight %}

    [[2. 3.]
     [4. 5.]
     [0. 0.]
     [0. 0.]]
    

#### hstack

syntax: *np.hstack((arrays_to_horizontal_stack))*
    
horizontal stacks all the arrays send in parameter


{% highlight python linenos %}
a=np.zeros(shape=var.shape)
horizontal=np.hstack((var,a))
print(horizontal)
{% endhighlight %}

    [[2. 3. 0. 0.]
     [4. 5. 0. 0.]]
    

#### ravel

syntax *np.ravel()*

flats out numpy array creating just a single row


{% highlight python linenos %}
flat=np.ravel(var)
print(flat)
{% endhighlight %}

    [2 3 4 5]
    

#### dot

syntax: *np.dot(array1,array2)*

find dot product of both arrays sends in the parameter.


{% highlight python linenos %}
a=np.array([[3,3],[5,2]])
dot_product=np.dot(var,a)
print(dot_product)
{% endhighlight %}

    [[21 12]
     [37 22]]
    

#### transpose

syntax: *np.transpose(array)*

finds the transpose of the array specified


{% highlight python linenos %}
np.transpose(var)
{% endhighlight %}




    array([[2, 4],
           [3, 5]])



#### linalg.inv

syntax: *np.linalg.inv(array)*

finds the inverse of array send in parameter


{% highlight python linenos %}
np.linalg.inv(var)
{% endhighlight %}




    array([[-2.5,  1.5],
           [ 2. , -1. ]])



#### linalg.det

syntax: *np.linalg.det(array)*

returns determinant of array send in parameter


{% highlight python linenos %}
np.linalg.det(var)
{% endhighlight %}


    -2.0
		
So thats what you needed to get started for everyday data science work. There are lot of other features but these can be learned as you progress your journey.
All the Best !!!

[IPython Notebook Link](https://github.com/tarun-bisht/blogs-notebooks/tree/master/numpy-crash-course)

### References
[numpy docs](https://docs.scipy.org/doc/numpy/reference/)