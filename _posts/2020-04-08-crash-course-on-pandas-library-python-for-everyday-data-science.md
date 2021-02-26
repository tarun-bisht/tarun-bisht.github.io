---
layout: writing
title:  crash course on pandas library python for everyday data science
date:   2020-04-08 20:46:50 +0530
category: Crash Course
tags: beginner python
comment: true
---
pandas is a fast, powerful, flexible and easy to use open-source data analysis and manipulation tool, built on top of the python programming language. This post acts as a cheat-sheet for using the pandas library in python for everyday data science. It contains some important functions and submodules of pandas library which are used day to day in data science and machine learning.
<!-- more -->

### Installing pandas

pandas can be installed easily by using pip. First, create a virtual environment and activate it. Then run the command below to install pandas.


{% highlight bash linenos %}
pip install pandas
{% endhighlight %} 

### Importing the library

Create a python project or open an existing project and import pandas using


{% highlight python linenos %}
import pandas
{% endhighlight %}

we can also use an alias pd to use pandas into the project by importing pandas as pd


{% highlight python linenos %}
import pandas as pd
{% endhighlight %}

### Reading and Visualizing CSV files

*data = pd.read_csv("path_to_csv_file")*


{% highlight python linenos %}
data = pd.read_csv("corona.csv")
{% endhighlight %}

this will read and load CSV file from memory and creates a pandas DataFrame object.

*data.head(number_of_columns)*

this will help to visualize how loaded data looks. It is stored in the DataFrame object which is pandas way to structure CSV data in the form of tables. ex:


{% highlight python linenos %}
data.head(5)  #shows first 5 rows of dataframe
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name of State / UT</th>
      <th>Total Confirmed cases *</th>
      <th>Cured/Discharged/Migrated</th>
      <th>Death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Andhra Pradesh</td>
      <td>19</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Andaman and Nicobar Islands</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Bihar</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Chandigarh</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Chhattisgarh</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Similarly, this will help to visualize the last part of data in DataFrame

*data.tail(numbers_of_columns)*


{% highlight python linenos %}
data.tail(5)
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name of State / UT</th>
      <th>Total Confirmed cases *</th>
      <th>Cured/Discharged/Migrated</th>
      <th>Death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>22</td>
      <td>Tamil Nadu</td>
      <td>50</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <td>23</td>
      <td>Telengana</td>
      <td>69</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>24</td>
      <td>Uttarakhand</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>Uttar Pradesh</td>
      <td>75</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <td>26</td>
      <td>West Bengal</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Getting shape or dimensions and size of loaded data

shape property gives out the shape or dimension of DataFrame as a tuple of the form (number_of_rows, number_of_columns) 


{% highlight python linenos %}
data.shape
{% endhighlight %}




    (27, 4)



size gives out the number of elements or number of unique cells in DataFrame.


{% highlight python linenos %}
data.size
{% endhighlight %}




    108



len on the DataFrame object gives out the number of rows in DataFrame.


{% highlight python linenos %}
len(data)
{% endhighlight %}




    27



### Retrieving data from DataFrame

*data['column_name']*

*data[['column_1','column_2','column_3']]*


{% highlight python linenos %}
 data['Death']
{% endhighlight %}




    0     0
    1     0
    2     1
    3     0
    4     0
    5     2
    6     0
    7     5
    8     0
    9     1
    10    2
    11    3
    12    1
    13    0
    14    2
    15    8
    16    0
    17    0
    18    0
    19    0
    20    1
    21    0
    22    1
    23    1
    24    0
    25    0
    26    1
    Name: Death, dtype: int64




{% highlight python linenos %}
data[['Name of State / UT','Death']]
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name of State / UT</th>
      <th>Death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Andhra Pradesh</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Andaman and Nicobar Islands</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Bihar</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Chandigarh</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Chhattisgarh</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Delhi</td>
      <td>2</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Goa</td>
      <td>0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Gujarat</td>
      <td>5</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Haryana</td>
      <td>0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Himachal Pradesh</td>
      <td>1</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Jammu and Kashmir</td>
      <td>2</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Karnataka</td>
      <td>3</td>
    </tr>
    <tr>
      <td>12</td>
      <td>Kerala</td>
      <td>1</td>
    </tr>
    <tr>
      <td>13</td>
      <td>Ladakh</td>
      <td>0</td>
    </tr>
    <tr>
      <td>14</td>
      <td>Madhya Pradesh</td>
      <td>2</td>
    </tr>
    <tr>
      <td>15</td>
      <td>Maharashtra</td>
      <td>8</td>
    </tr>
    <tr>
      <td>16</td>
      <td>Manipur</td>
      <td>0</td>
    </tr>
    <tr>
      <td>17</td>
      <td>Mizoram</td>
      <td>0</td>
    </tr>
    <tr>
      <td>18</td>
      <td>Odisha</td>
      <td>0</td>
    </tr>
    <tr>
      <td>19</td>
      <td>Puducherry</td>
      <td>0</td>
    </tr>
    <tr>
      <td>20</td>
      <td>Punjab</td>
      <td>1</td>
    </tr>
    <tr>
      <td>21</td>
      <td>Rajasthan</td>
      <td>0</td>
    </tr>
    <tr>
      <td>22</td>
      <td>Tamil Nadu</td>
      <td>1</td>
    </tr>
    <tr>
      <td>23</td>
      <td>Telengana</td>
      <td>1</td>
    </tr>
    <tr>
      <td>24</td>
      <td>Uttarakhand</td>
      <td>0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>Uttar Pradesh</td>
      <td>0</td>
    </tr>
    <tr>
      <td>26</td>
      <td>West Bengal</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



sort and return DataFrame column specified


{% highlight python linenos %}
data.sort_values(['Total Confirmed cases *']) #sort and return DataFrame column
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name of State / UT</th>
      <th>Total Confirmed cases *</th>
      <th>Cured/Discharged/Migrated</th>
      <th>Death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>19</td>
      <td>Puducherry</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>17</td>
      <td>Mizoram</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>16</td>
      <td>Manipur</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>18</td>
      <td>Odisha</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Himachal Pradesh</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Goa</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Chhattisgarh</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>24</td>
      <td>Uttarakhand</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Chandigarh</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Andaman and Nicobar Islands</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Bihar</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>13</td>
      <td>Ladakh</td>
      <td>13</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Andhra Pradesh</td>
      <td>19</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>26</td>
      <td>West Bengal</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Jammu and Kashmir</td>
      <td>31</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <td>14</td>
      <td>Madhya Pradesh</td>
      <td>33</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Haryana</td>
      <td>33</td>
      <td>17</td>
      <td>0</td>
    </tr>
    <tr>
      <td>20</td>
      <td>Punjab</td>
      <td>38</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>22</td>
      <td>Tamil Nadu</td>
      <td>50</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Delhi</td>
      <td>53</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <td>21</td>
      <td>Rajasthan</td>
      <td>57</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Gujarat</td>
      <td>58</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <td>23</td>
      <td>Telengana</td>
      <td>69</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>25</td>
      <td>Uttar Pradesh</td>
      <td>75</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Karnataka</td>
      <td>80</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <td>15</td>
      <td>Maharashtra</td>
      <td>193</td>
      <td>25</td>
      <td>8</td>
    </tr>
    <tr>
      <td>12</td>
      <td>Kerala</td>
      <td>194</td>
      <td>19</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### Using loc and iloc

pandas DataFrame provides powerful properties loc and iloc to retrieve data from DataFrame. This is the most widely used technique if we have lots of features to retrieve.

loc selects things by using the label

loc has following valid inputs -

1. A single label, e.g. 5 or 'a' (Note that 5 is interpreted as a label of the index. This use is not an integer position along with the index.).

2. A list or array of labels ['a', 'b', 'c'].

3. A slice object with labels' a ' : ' f '(Note that contrary to usual python slices, both the start and the stop are included.)

*data.loc[row_needed,column_needed]   #loc with input type 1*

*data.loc[[row_needed(multiple)],column_needed]   #loc with input type 2*

*data.loc[row_range_needed,column_range_needed]   #loc with input type 3*

Example:


{% highlight python linenos %}
data.loc[:,'Total Confirmed cases *'] #selects all rows from column Total Confirmed cases
{% endhighlight %}




    0      19
    1       9
    2      11
    3       8
    4       7
    5      53
    6       5
    7      58
    8      33
    9       3
    10     31
    11     80
    12    194
    13     13
    14     33
    15    193
    16      1
    17      1
    18      3
    19      1
    20     38
    21     57
    22     50
    23     69
    24      7
    25     75
    26     19
    Name: Total Confirmed cases *, dtype: int64




{% highlight python linenos %}
data.loc[:,['Total Confirmed cases *','Death']]
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total Confirmed cases *</th>
      <th>Death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>19</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>53</td>
      <td>2</td>
    </tr>
    <tr>
      <td>6</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>58</td>
      <td>5</td>
    </tr>
    <tr>
      <td>8</td>
      <td>33</td>
      <td>0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <td>10</td>
      <td>31</td>
      <td>2</td>
    </tr>
    <tr>
      <td>11</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <td>12</td>
      <td>194</td>
      <td>1</td>
    </tr>
    <tr>
      <td>13</td>
      <td>13</td>
      <td>0</td>
    </tr>
    <tr>
      <td>14</td>
      <td>33</td>
      <td>2</td>
    </tr>
    <tr>
      <td>15</td>
      <td>193</td>
      <td>8</td>
    </tr>
    <tr>
      <td>16</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>17</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>18</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>19</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>20</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <td>21</td>
      <td>57</td>
      <td>0</td>
    </tr>
    <tr>
      <td>22</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <td>23</td>
      <td>69</td>
      <td>1</td>
    </tr>
    <tr>
      <td>24</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>75</td>
      <td>0</td>
    </tr>
    <tr>
      <td>26</td>
      <td>19</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight python linenos %}
data.loc[:,'Total Confirmed cases *':'Death'] #( including death column [refer input type 3])
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total Confirmed cases *</th>
      <th>Cured/Discharged/Migrated</th>
      <th>Death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>19</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>53</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <td>6</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>58</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <td>8</td>
      <td>33</td>
      <td>17</td>
      <td>0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>10</td>
      <td>31</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <td>11</td>
      <td>80</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <td>12</td>
      <td>194</td>
      <td>19</td>
      <td>1</td>
    </tr>
    <tr>
      <td>13</td>
      <td>13</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>14</td>
      <td>33</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>15</td>
      <td>193</td>
      <td>25</td>
      <td>8</td>
    </tr>
    <tr>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>17</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>18</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>19</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>20</td>
      <td>38</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>21</td>
      <td>57</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>22</td>
      <td>50</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <td>23</td>
      <td>69</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>24</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>75</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <td>26</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**note: using loc slices contrary to usual python slices, both the start and the stop is included, when present in the index ie.. in ' a ' : ' c ' both ' a ' and ' c ' row are inclusive.**

iloc selects rows and columns using integer positions

iloc has the following valid inputs -

1: An integer e.g. 5.

2: A list or array of integers [ 4, 3, 0 ].

3: A slice object with ints 1 : 7.

4: A boolean array.

*data.iloc[row_index_needed,column_index_needed] #iloc with type 1*

*data.iloc[[row_index_needed(multiple)],column_index_needed] #iloc with type 2*

*data.iloc[row_index_range_needed,column_index_range_needed] #iloc with type 3*

Example:



{% highlight python linenos %}
data.iloc[:,1]   #selects all rows of column with index 1
{% endhighlight %}




    0      19
    1       9
    2      11
    3       8
    4       7
    5      53
    6       5
    7      58
    8      33
    9       3
    10     31
    11     80
    12    194
    13     13
    14     33
    15    193
    16      1
    17      1
    18      3
    19      1
    20     38
    21     57
    22     50
    23     69
    24      7
    25     75
    26     19
    Name: Total Confirmed cases *, dtype: int64




{% highlight python linenos %}
data.iloc[:,[1,3]]   #selects every row of column with index 1 and 4
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total Confirmed cases *</th>
      <th>Death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>19</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>53</td>
      <td>2</td>
    </tr>
    <tr>
      <td>6</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>58</td>
      <td>5</td>
    </tr>
    <tr>
      <td>8</td>
      <td>33</td>
      <td>0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <td>10</td>
      <td>31</td>
      <td>2</td>
    </tr>
    <tr>
      <td>11</td>
      <td>80</td>
      <td>3</td>
    </tr>
    <tr>
      <td>12</td>
      <td>194</td>
      <td>1</td>
    </tr>
    <tr>
      <td>13</td>
      <td>13</td>
      <td>0</td>
    </tr>
    <tr>
      <td>14</td>
      <td>33</td>
      <td>2</td>
    </tr>
    <tr>
      <td>15</td>
      <td>193</td>
      <td>8</td>
    </tr>
    <tr>
      <td>16</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>17</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>18</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>19</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>20</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <td>21</td>
      <td>57</td>
      <td>0</td>
    </tr>
    <tr>
      <td>22</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <td>23</td>
      <td>69</td>
      <td>1</td>
    </tr>
    <tr>
      <td>24</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>75</td>
      <td>0</td>
    </tr>
    <tr>
      <td>26</td>
      <td>19</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight python linenos %}
data.iloc[:,1:3]  #selects every row of column with index 1 and 2 only
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total Confirmed cases *</th>
      <th>Cured/Discharged/Migrated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>19</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>53</td>
      <td>6</td>
    </tr>
    <tr>
      <td>6</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>58</td>
      <td>1</td>
    </tr>
    <tr>
      <td>8</td>
      <td>33</td>
      <td>17</td>
    </tr>
    <tr>
      <td>9</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>10</td>
      <td>31</td>
      <td>1</td>
    </tr>
    <tr>
      <td>11</td>
      <td>80</td>
      <td>5</td>
    </tr>
    <tr>
      <td>12</td>
      <td>194</td>
      <td>19</td>
    </tr>
    <tr>
      <td>13</td>
      <td>13</td>
      <td>3</td>
    </tr>
    <tr>
      <td>14</td>
      <td>33</td>
      <td>0</td>
    </tr>
    <tr>
      <td>15</td>
      <td>193</td>
      <td>25</td>
    </tr>
    <tr>
      <td>16</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>17</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>18</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>19</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>20</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <td>21</td>
      <td>57</td>
      <td>3</td>
    </tr>
    <tr>
      <td>22</td>
      <td>50</td>
      <td>4</td>
    </tr>
    <tr>
      <td>23</td>
      <td>69</td>
      <td>1</td>
    </tr>
    <tr>
      <td>24</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <td>25</td>
      <td>75</td>
      <td>11</td>
    </tr>
    <tr>
      <td>26</td>
      <td>19</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight python linenos %}
data.iloc[0:3,:]  #selects rows 0 to 2 and every column
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name of State / UT</th>
      <th>Total Confirmed cases *</th>
      <th>Cured/Discharged/Migrated</th>
      <th>Death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Andhra Pradesh</td>
      <td>19</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Andaman and Nicobar Islands</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Bihar</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



both iloc and loc also take callables as valid input. Callables are functions that return row and column needed to select. This is useful when we have to filter some information before selecting it.

*data.loc[callable_function]*

*data.iloc[callable_function]*

Example


{% highlight python linenos %}
data.loc[lambda data:data["Name of State / UT"] == 'Uttarakhand'] #selects all rows and columns with country India
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name of State / UT</th>
      <th>Total Confirmed cases *</th>
      <th>Cured/Discharged/Migrated</th>
      <th>Death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>24</td>
      <td>Uttarakhand</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight python linenos %}
data.iloc[lambda x:x.index % 2 == 0] #selects all rows and columns with even index
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name of State / UT</th>
      <th>Total Confirmed cases *</th>
      <th>Cured/Discharged/Migrated</th>
      <th>Death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Andhra Pradesh</td>
      <td>19</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Bihar</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Chhattisgarh</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Goa</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Haryana</td>
      <td>33</td>
      <td>17</td>
      <td>0</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Jammu and Kashmir</td>
      <td>31</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <td>12</td>
      <td>Kerala</td>
      <td>194</td>
      <td>19</td>
      <td>1</td>
    </tr>
    <tr>
      <td>14</td>
      <td>Madhya Pradesh</td>
      <td>33</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <td>16</td>
      <td>Manipur</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>18</td>
      <td>Odisha</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>20</td>
      <td>Punjab</td>
      <td>38</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>22</td>
      <td>Tamil Nadu</td>
      <td>50</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <td>24</td>
      <td>Uttarakhand</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>26</td>
      <td>West Bengal</td>
      <td>19</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Plotting a DataFrame

We can also plot a DataFrame in the graph which uses Matplotlib library to plot graph. This can be used to plot simple graphs without needing to write plot code.

It plot graphs of graph type specified

Following types of graphs are valid in pandas:

1. bar or barh for bar plots

2. hist for histogram

3. box for boxplot

4. kde or density for density plots

5. area for area plots

6. scatter for scatter plots

7. hexbin for hexagonal bin plots

8. pie for pie plots

syntax: *data.plot(kind='graph_type')*


{% highlight python linenos %}
data.plot(x ='Name of State / UT', y='Total Confirmed cases *',kind='bar')
{% endhighlight %}




    <matplotlib.axes._subplots.AxesSubplot at 0x2272464b148>




![png](https://storage.googleapis.com/tarun-bisht.appspot.com/blogs/pandas_bar_graph3dd3074ff7cb5424)


### Concatenating DataFrame with another

We can concatenate DataFrame using pd.concat method

syntax: *pd.concat([dataframe1,dataframe2,….],axis= 0 or 1)*

pass list of DataFrame to which are needed to concatenate and axis from where to concatenate


{% highlight python linenos %}
data1 = {'Enrollment No.': [1,2,3,4,5,6,7,8,9,10],
        'Score': [1500,1520,1525,1523,1515,1540,1545,1560,1555,1565]
       }
data2 = {'Class': [0,1,2,1,1,2,0,0,1,2],
        'Place': [1,3,1,2,5,1,5,4,2,1]
       }
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

df=pd.concat([df1,df2],axis=1)
df.head(5)
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Enrollment No.</th>
      <th>Score</th>
      <th>Class</th>
      <th>Place</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1500</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>1520</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>1525</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>1523</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>1515</td>
      <td>1</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



### Checking null values in column

pd.isna function is used to check for missing values like NaN or blank etc.

syntax: *pd.isna(data[column_name])*


{% highlight python linenos %}
pd.isna(data['Death'])
{% endhighlight %}




    0     False
    1     False
    2     False
    3     False
    4     False
    5     False
    6     False
    7     False
    8     False
    9     False
    10    False
    11    False
    12    False
    13    False
    14    False
    15    False
    16    False
    17    False
    18    False
    19    False
    20    False
    21    False
    22    False
    23    False
    24    False
    25    False
    26    False
    Name: Death, dtype: bool
		
		
So thats what you needed to get started for everyday data science work. There are lot of other features but these can be learned as you progress your journey.
All the Best !!!

[IPython Notebook Link](https://github.com/tarun-bisht/blogs-notebooks/tree/master/numpy-crash-course)

### References
[pandas docs](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)