---
layout: post
title: Risk measurement and creating the analysis module
subtitle: Getting the basics out of the way II
published: true
---

The second blog-post will provide more functions for risk measurement. At the end of this post, we will combine everything to get a basic analysis tool for return series. 
Some of the functions will include functions which were defined in the first blog post.
For more information on the monthly industry returns data and the returns- and distribution functions see als [blog post I](https://fallers.github.io/01.14.2021-first_post/)

# General functions part II

```python
# import packages
import pandas as pd
import numpy as np
from scipy.stats import norm

# load data
ind = pd.read_csv("ind30_m_valueweight_rets.CSV", header=0, index_col=0)/100
ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
ind = ind[:"2018"]
ind.columns = ind.columns.str.strip()
```

### Drawdown
The drawdown function provides a time-series, as well as local and global drawdown for a given pandas time series: 

```python
def drawdown(return_series: pd.Series):
    """Takes return series
       returns DataFrame with columns
       wealth index, 
       previous peaks, 
       the percentage drawdown
    """
    wealth_index = 100*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})
```

We can easily look up the series for one of our industry portfolios and check for the drawdowns in this time or look up the maximum dardown and time of this drawdown.

```python
drawdown(ind['Smoke'])['Drawdown'].plot.line()
```




    <AxesSubplot:>




![png](/assets/img/2021-01-15-second_post_files/output_5_1.png)


```python
# take time series of "Smoke" from the DataFrame for the drawdown function
round(drawdown(ind['Smoke'])['Drawdown'].min(),4)*100
```




    -59.88



```python
# maximum drawdown
drawdown(ind['Smoke'])['Drawdown'].idxmin()
```




    Period('2000-02', 'M')



The maximum drawdown until end of 2018 for Smoke was -59.88% on 2000-02.

### Semideviation
Identify the volatility of only negative returns since we don't want to specify upside movement as risk

```python
def semideviation(r):
    """
    Returns negative (semi)deviation of returns series
    r must be a Series or a DataFrame
    """
    if isinstance(r, pd.Series):
        # mask to identify the negative returns 
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
```

We get the following downside-deviations in percent:

```python
semideviation(ind[['Smoke', 'Fin']])*100
```




    Smoke    3.947727
    Fin      5.137144
    dtype: float64



## Value at Risk
We want to estimate, how much a investments might lose given a predefined probability, and given normal market conditions, in a set time period
There are different ways to measure VaR

### Historic VaR
We want the lowest 5% of returns in our historic dataset. 
We will leverage the already existing np.percentile

```python
def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    If not explicitely defined, level = 5%
    Takes pandas DataFrame or Series
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        #take level-percentile of return series
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
```

```python
var_historic(ind[['Smoke', 'Food']]['1995':'2018'])*100
```




    Smoke    9.947
    Food     5.774
    dtype: float64



Based on 5 years of data from 1995 - 2018, in 5% of all cases, loose at least appr. 9,95% in Smoke indursties investment

### Conditional Value at Risk / expeced shortfall
Provides an average of returns in the VaR-case

```python
def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR 
    Takes pandas DataFrame or Series
    """
    if isinstance(r, pd.Series):
        # mask to identify all returns which are below the threshold 
        is_beyond = r <= -var_historic(r, level=level)
        # provide mean of the returns which are below the threshold
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
```

```python
cvar_historic(ind[['Smoke', 'Food']]['1995':'2018'])*100
```




    Smoke    15.218000
    Food      7.939333
    dtype: float64



On average the loss in the Smoke industry is 15,22% based on the conditions listed in the historic VaR section

### Parametric Gaussian VaR
Computes VaR based on mean and standard deviation, adjusting the result by z-score based on predefined level. Normal distribution is assumed. 
With [Cornish-Fisher modification](https://en.wikipedia.org/wiki/Cornish%E2%80%93Fisher_expansion), we can modify the z-score based on historic skewness and kurtosis 


![png](/assets/img/2021-01-15-second_post_files/output-cornish-fisher.png)



```python
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    Takes pandas DataFrame or Series
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis function from blog post I
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))
```

```python
var_gaussian(ind[['Smoke', 'Food']]['1995':'2018'])*100
```




    Smoke    9.865625
    Food     5.272661
    dtype: float64



```python
var_gaussian(r=ind[['Smoke', 'Food']]['1995':'2018'], modified=True)*100
```




    Smoke    9.885074
    Food     5.485750
    dtype: float64



Both VaR-Versions are pretty close to our historic VaR results

We can plot a comparison for these VaR-approaches for some more industries:

```python
data = ind[['Smoke', 'Food', 'Fin', 'Books', 'Whlsl']]['1995':'2018']

var_table = [var_gaussian(data), 
             var_gaussian(data, modified=True), 
             var_historic(data)]
var_comparison = pd.concat(var_table, axis=1)
var_comparison.columns=['Gaussian', 'Cornish-Fisher', 'Historic']
var_comparison.plot.bar(title="Different industries: VaR at 5%")
```




    <AxesSubplot:title={'center':'Different industries: VaR at 5%'}>




![png](/assets/img/2021-01-15-second_post_files/output_28_1.png)


```python
def summary_stats(r, riskfree_rate=0.02):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    Annual riskfree rate assumed 2%
    Takes pandas DataFrame or series
    """
    # utilize aggregate method to use functions over all columns
    ann_r = r.aggregate(annualize_ret, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    # anonymous function to find minimum of Drawdown-column in output
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
        
    }, index=r.columns)
```

```python
summary_stats(ind[['Smoke', 'Fin', 'Food']])
```




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
      <th>Annualized Return</th>
      <th>Annualized Vol</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
      <th>Cornish-Fisher VaR (5%)</th>
      <th>Historic CVaR (5%)</th>
      <th>Sharpe Ratio</th>
      <th>Max Drawdown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Smoke</th>
      <td>0.122324</td>
      <td>0.201206</td>
      <td>0.003873</td>
      <td>6.238037</td>
      <td>0.080292</td>
      <td>0.122454</td>
      <td>0.499260</td>
      <td>-0.598755</td>
    </tr>
    <tr>
      <th>Fin</th>
      <td>0.097000</td>
      <td>0.234567</td>
      <td>0.517500</td>
      <td>14.666227</td>
      <td>0.075073</td>
      <td>0.152759</td>
      <td>0.322167</td>
      <td>-0.921246</td>
    </tr>
    <tr>
      <th>Food</th>
      <td>0.106606</td>
      <td>0.163979</td>
      <td>0.047540</td>
      <td>9.687574</td>
      <td>0.061204</td>
      <td>0.103661</td>
      <td>0.518538</td>
      <td>-0.722459</td>
    </tr>
  </tbody>
</table>
</div>



## Building the Module fin_pack
We can easily manage all the functions from blog post 1 and this blog post if we integrate them in one module and import this module for future usage.
All functions will be saved in one .py file which we will name fin_pack.Remember to include all dependencies like pandas, numpy etc.

We can import the package  fin_pack at the beginning of our work to get access to the functions. in the module, e.g. **import fin_pack as fipa** allows us to use the summary function and all other functions as **fipa.summary_stats(ind[['Smoke', 'Fin', 'Food']])**

When actively working, updating and testing the module with Jupyter Notebook, the fllowing magic commands can be added in the notebook to get updated results when changing the package:
**%load_ext autoreload**

**%autoreload 2**


```python
import fin_pack as fipa
```

```python
fipa.summary_stats(ind[['Smoke', 'Fin', 'Food']])
```




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
      <th>Annualized Return</th>
      <th>Annualized Vol</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
      <th>Cornish-Fisher VaR (5%)</th>
      <th>Historic CVaR (5%)</th>
      <th>Sharpe Ratio</th>
      <th>Max Drawdown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Smoke</th>
      <td>0.122324</td>
      <td>0.201206</td>
      <td>0.003873</td>
      <td>6.238037</td>
      <td>0.080292</td>
      <td>0.122454</td>
      <td>0.499260</td>
      <td>-0.598755</td>
    </tr>
    <tr>
      <th>Fin</th>
      <td>0.097000</td>
      <td>0.234567</td>
      <td>0.517500</td>
      <td>14.666227</td>
      <td>0.075073</td>
      <td>0.152759</td>
      <td>0.322167</td>
      <td>-0.921246</td>
    </tr>
    <tr>
      <th>Food</th>
      <td>0.106606</td>
      <td>0.163979</td>
      <td>0.047540</td>
      <td>9.687574</td>
      <td>0.061204</td>
      <td>0.103661</td>
      <td>0.518538</td>
      <td>-0.722459</td>
    </tr>
  </tbody>
</table>
</div>



This will close out the basic functions to analyze return series. 
for the next posts we will take a look at portfolio optimization :)
