---
layout: post
title: Basic returns and distibution functions
subtitle: Getting the basics out of the way I
published: True
---

The first two blog posts will serve as basis for the upcoming posts and topics to get some of the very basics out of the way. 
General Python and finance knowledge is assumed. There are also various sources for Pythong in general and finance-related topics in specific online, some can be found in the resources of this blog. 

Let's start with the very basics which we will be using in the following posts on more interesting topics: 
* data and data-source
* packages and basic functions we will be using 

Some of the functions build on each other. The functions will be used in later posts with a short notice of this post.
At the end of the two blog-posts, we will pack all these functions in one module so we can access it for further work.

```python
# packages used for these functions
import pandas as pd
import numpy as np
import scipy.stats
```
# Data-source

We will be using part of [Kenneth R. Frenchs data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), specifically the 30 Industry Portfolios.csv file.
The file includes several datasets one after the next in the samle sheet based on weighting schemes. We will be working with the value-weighted dataset which begins with row 12 for the indusries and ends based in the last update somewhere in 202011 or row 1145+. simply copy this dataset, first row being the industries, in a separate csv-file with the name ind30_m_valueweight_rets.csv.  

We will now load and adjust the data to make it more convinient to work with and take a look at the data. 
the file path may need to be adjusted depending where the .csv fiel is stored
For now we will also only take the data up until 2018.

```python
# read in csv as pandas DataFrame, define header and index-column
# folder path needs to be adjusted
ind = pd.read_csv("ind30_m_valueweight_rets.CSV", header=0, index_col=0)/100
# define index as month-date
ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
#reduce data until year 2018 for ease of use
ind = ind[:"2018"]
# strip trailing spaces
ind.columns = ind.columns.str.strip()

# check columns to see the industries
print(ind.columns)

#take a look at the data
print(ind.shape)
print(ind.head())
```

    Index(['Food', 'Beer', 'Smoke', 'Games', 'Books', 'Hshld', 'Clths', 'Hlth',
           'Chems', 'Txtls', 'Cnstr', 'Steel', 'FabPr', 'ElcEq', 'Autos', 'Carry',
           'Mines', 'Coal', 'Oil', 'Util', 'Telcm', 'Servs', 'BusEq', 'Paper',
           'Trans', 'Whlsl', 'Rtail', 'Meals', 'Fin', 'Other'],
          dtype='object')
    (1110, 30)
               Food    Beer   Smoke   Games   Books   Hshld   Clths    Hlth  \
    1926-07  0.0056 -0.0519  0.0129  0.0293  0.1097 -0.0048  0.0808  0.0177   
    1926-08  0.0259  0.2703  0.0650  0.0055  0.1001 -0.0358 -0.0251  0.0425   
    1926-09  0.0116  0.0402  0.0126  0.0658 -0.0099  0.0073 -0.0051  0.0069   
    1926-10 -0.0306 -0.0331  0.0106 -0.0476  0.0947 -0.0468  0.0012 -0.0057   
    1926-11  0.0635  0.0729  0.0455  0.0166 -0.0580 -0.0054  0.0187  0.0542   
    
              Chems   Txtls  ...   Telcm   Servs   BusEq   Paper   Trans   Whlsl  \
    1926-07  0.0814  0.0039  ...  0.0083  0.0922  0.0206  0.0770  0.0193 -0.2379   
    1926-08  0.0550  0.0797  ...  0.0217  0.0202  0.0439 -0.0238  0.0488  0.0539   
    1926-09  0.0533  0.0230  ...  0.0241  0.0225  0.0019 -0.0554  0.0005 -0.0787   
    1926-10 -0.0476  0.0100  ... -0.0011 -0.0200 -0.0109 -0.0508 -0.0264 -0.1538   
    1926-11  0.0520  0.0310  ...  0.0163  0.0377  0.0364  0.0384  0.0160  0.0467   
    
              Rtail   Meals     Fin   Other  
    1926-07  0.0007  0.0187  0.0037  0.0520  
    1926-08 -0.0075 -0.0013  0.0446  0.0676  
    1926-09  0.0025 -0.0056 -0.0123 -0.0386  
    1926-10 -0.0220 -0.0411 -0.0516 -0.0849  
    1926-11  0.0652  0.0433  0.0224  0.0400  
    
    [5 rows x 30 columns]


# General functions part I
## Returns and its distribution
Following is the first part of the general functions to analyse return series.
Part I will consist of functions related to returns and its distribution.
Part II in the follow-up blogpost will provide more content for measuring risk in particular. 

### Annualize returns

```python
def annualize_ret(r, periods_per_year):
    """
    Annualizes returns of return series
    Requires return series
    and periods per year
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1
```

```python
annualize_ret(ind['Smoke'],12)*100
```




    12.232396767074283



The annualized return of the Smoke industry data was appr. 12.23%.

### Annualize volatility
volatility as standard deviation, annualized

```python
def annualize_vol(r, periods_per_year):
    """
    Annualizes volatility of return series
    Requires return series
    and periods per year
    """
    return r.std()*np.sqrt(periods_per_year)
```

```python
ann_vol_smoke = annualize_vol(ind['Smoke'],12)
```

The annualized volatility of the Smoke industry data was appr. 20.12%.

### Sharpe Ratio
Calculates the sharpe ratio on an annual basis. 
Unrealistic assumption to have the same riskfree rate for the complete time horizon but we are starting small :)

```python
def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    Requires return series,  
    periods per year
    and an annual risk-free rate.
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_ret(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol
```

```python
sharpe_ratio(r=ind['Smoke'], riskfree_rate=0.02, periods_per_year=12)
```




    0.4992601550073793



The annualized share ratio of the Smoke industry assuming a riskfree rate of 2% was appr. 0,5.

### Skewness
Negative skew results in more negative returns as would be expected cmopared to a normal distribution

```python
# alternatively scipy can be used: scipy.stats.skew()
def skewness(r):
    """
    Computes skewness of Series or DataFrame
    Returns a float or a Series depending n input
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

```

```python
skewness(ind['Smoke'])
```




    0.0038726766269701783



### Kurtosis
Relates to "fatness" of tails compared to normal distributuion. Fatter tails describe more extreme outcomes as would be expected compared to a normal distribution as can be often seen with stock investments. 
Normal distribution has a kurtosis of 3, so a kurtosis > 3 related to fat tails

```python
# alternatively scipy can be used: scipy.stats.kurtosis()
def kurtosis(r):
    """
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4
```

```python
kurtosis(ind['Smoke'])
# well above 3 ;)
```




    6.238037198836751



### Test for Normality: Jarque-Bera
The [Jarque-Bera test](https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test) to test if returns series is normally distributed. 
Provides p-value for acceptance of test for normal distribution

```python
def is_normal(r, level=0.01):
    """
    Takes pandas DataFrame or Series
    Applies the Jarque-Bera test to determine if a Series is normally distributed
    Default level = 1%
    Returns True if the null-hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level
```

```python
is_normal(ind[['Smoke', 'Fin']])
```




    Smoke    False
    Fin      False
    dtype: bool



No surprise, the return series is not normally distirbuted (as already shown with the kurtosis). 
We can also add additional columns / industries for this function thanks to the aggregate method

The next blog-post will provide additional functions for risk-measurement and combine the results of boths posts in one function to analyse return series!
