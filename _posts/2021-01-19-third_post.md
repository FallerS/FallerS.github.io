---
layout: post
title: Portfolio Construction and Optimization
subtitle: Efficient frontier and CML
published: true
---

This blog post with explain some of the general portfolio construction topics. How should the investments optimally be weighted in the contruction step? Which role does the covariance between these investments play and how can we use it to our advantage?

Some knowledge of these topics is assumed and implement them in a Python framework. Let's start!

## Importing and checking the dataset
See also [blog post I](https://fallers.github.io/01.14.2021-first_post/) for details on the data. 

```python
# import modules needed
import pandas as pd
import numpy as np
# import the prepared module from blog post 1 & 2
import fin_pack as fipa

# load data
ind = pd.read_csv("ind30_m_valueweight_rets.CSV", header=0, index_col=0)/100
ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
ind = ind[:"2018"]
ind.columns = ind.columns.str.strip()

#short reminder of the dataset
print(ind.shape)
print(ind.columns)
print(ind.head())
```

    (1110, 30)
    Index(['Food', 'Beer', 'Smoke', 'Games', 'Books', 'Hshld', 'Clths', 'Hlth',
           'Chems', 'Txtls', 'Cnstr', 'Steel', 'FabPr', 'ElcEq', 'Autos', 'Carry',
           'Mines', 'Coal', 'Oil', 'Util', 'Telcm', 'Servs', 'BusEq', 'Paper',
           'Trans', 'Whlsl', 'Rtail', 'Meals', 'Fin', 'Other'],
          dtype='object')
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


## Efficient frontier
We want to find the combination of assets (here: industry portfolios) that grant the highest exp. return for a given risk level or the lowest risk for a defined exp. return level.
We will need expected returns and a covariance matrix to get the efficient frontier. We will simply work with historical results so we don't have to approach the estaimation of input parameters and its associated problems for now. 

```python
# create historicals return series which we will use as expected returns for this post
er = fipa.annualize_ret(ind["1995":"2018"], 12)
er.sort_values().plot.bar(title='annualized returns of industry portfolios 1995 - 2018')
```




    <AxesSubplot:title={'center':'annualized returns of industry portfolios 1995 - 2018'}>




![png](/assets/img/2021-01-19-third_post_files/output_4_1.png)


```python
# get historical covariance matrix of industry returns we will use in this post
cov = ind["1995":"2018"].cov()
cov.shape
```




    (30, 30)



First organize a framework for portfolio returns and volatility

```python
def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    #leveraging @ for matrix multiplication
    return weights.T @ returns
```

```python
def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5
```

We will add these functions to our fipa module from the last blog post. 

We now want the weightings for a given set of investments (industry portfolios) which minimizes the risk for a given exp. portfolio return
We will use scypis [optimizer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) to do this. 

```python
from scipy.optimize import minimize


def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    Takes a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    # specification of init guess is notimportant, another intial guess to start minimization also works
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples! First bound defines investment weighting between 0 and 1, no shorting!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1 # weights of all investments sum to 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er) # specifies that the targetet return for risk minimization equals the specified return
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x


```

Lets do a short testcase

```python
er[['Smoke', 'Fin']]
```




    Smoke    0.130613
    Fin      0.096563
    dtype: float64



```python
cov.loc[['Smoke', 'Fin'], ['Smoke', 'Fin']]
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
      <th>Smoke</th>
      <th>Fin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Smoke</th>
      <td>0.004590</td>
      <td>0.001035</td>
    </tr>
    <tr>
      <th>Fin</th>
      <td>0.001035</td>
      <td>0.003110</td>
    </tr>
  </tbody>
</table>
</div>



```python
# only take Smoke and Fin industries, target return of 0,115
minimize_vol(target_return=0.115, er=er[['Smoke', 'Fin']], cov=cov.loc[['Smoke', 'Fin'], ['Smoke', 'Fin']])
```




    array([0.5414673, 0.4585327])



Targeting a return of 11,5%, we would weight our portfolio with ca. 54% of Smoke and 46% Financials industries

Next, we want to use the minimizer for a range of returns to get the optimal weights:

```python
# evenly space the target returns of smoke and Fin
target_rs = np.linspace(start=er[['Smoke', 'Fin']].min(), stop=er[['Smoke', 'Fin']].max(), num=10)  

# use minimizer for target returns to get optimal weights 
weights = [minimize_vol(target_return, er[['Smoke', 'Fin']], cov.loc[['Smoke', 'Fin'], ['Smoke', 'Fin']]) for target_return in target_rs]

# generate DataFrame with results
df = pd.DataFrame({'target_rets': target_rs,'weights': weights})
df['weights'] = [np.round(w, 4) for w in df['weights']]
df['target_rets'] = round(df['target_rets'],4)
df
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
      <th>target_rets</th>
      <th>weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0966</td>
      <td>[0.0, 1.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1003</td>
      <td>[0.1111, 0.8889]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.1041</td>
      <td>[0.2222, 0.7778]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.1079</td>
      <td>[0.3333, 0.6667]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.1117</td>
      <td>[0.4444, 0.5556]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.1155</td>
      <td>[0.5556, 0.4444]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.1193</td>
      <td>[0.6667, 0.3333]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.1230</td>
      <td>[0.7778, 0.2222]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.1268</td>
      <td>[0.8889, 0.1111]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.1306</td>
      <td>[1.0, 0.0]</td>
    </tr>
  </tbody>
</table>
</div>



These results show on a 2-asset case the weighting development when targeting a range of returns.
The weighting results noticeably switch in extreme results very fast as the target returns change from ca 10% to 13%. This tendency to extreme weightings with only very small changes in (expected) returns will be a significant problem. 

For now lets formalize our results for future work and illustrate them as a graph.

```python
# based on our lines of code before:
def optimal_weights(n_points, er, cov, df=False):
    """
    Returns a list of weights for evenly spaced target returns
    Takes number of target returns, expected returns, covariance
    """
    target_rs = np.linspace(er.min(), er.max(), n_points) #spaces out the target-returns evenly between the minimum and maximum returns to get weightings for every return 
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    if df:
        # generate DataFrame with results
        df = pd.DataFrame({'target_rets': target_rs,'weights': weights})
        df['weights'] = [np.round(w, 4) for w in df['weights']]
        df['target_rets'] = round(df['target_rets'],4)
        return df
    else:
        return weights
```

```python
# plot efficient frontier based on functions in this post
def plot_ef(n_points, er, cov, style='.-', legend=False):
    """
    Plots the multi-asset efficient frontier
    Takes number of target returns, exp. returns, covariance, riskfree rate
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    return ax
```

```python
# relevant industries: let's try more industries
i = ['Hlth', 'Rtail', 'Beer']
#hist. annual returns as expt. returns
er[i]
```




    Hlth     0.117599
    Rtail    0.115921
    Beer     0.102976
    dtype: float64



```python
# hist. covariance
cov.loc[i, i]
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
      <th>Hlth</th>
      <th>Rtail</th>
      <th>Beer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Hlth</th>
      <td>0.001772</td>
      <td>0.000992</td>
      <td>0.001049</td>
    </tr>
    <tr>
      <th>Rtail</th>
      <td>0.000992</td>
      <td>0.002398</td>
      <td>0.000883</td>
    </tr>
    <tr>
      <th>Beer</th>
      <td>0.001049</td>
      <td>0.000883</td>
      <td>0.002218</td>
    </tr>
  </tbody>
</table>
</div>



```python
optimal_weights(10,er[i], cov.loc[i,i], df=True)
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
      <th>target_rets</th>
      <th>weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.1030</td>
      <td>[0.0, 0.0, 1.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1046</td>
      <td>[0.0, 0.1255, 0.8745]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.1062</td>
      <td>[0.0562, 0.1875, 0.7563]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.1079</td>
      <td>[0.1522, 0.2047, 0.6432]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.1095</td>
      <td>[0.2211, 0.2523, 0.5266]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.1111</td>
      <td>[0.3331, 0.2513, 0.4156]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.1127</td>
      <td>[0.4254, 0.2726, 0.3021]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.1143</td>
      <td>[0.5177, 0.2938, 0.1885]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.1160</td>
      <td>[0.6101, 0.3149, 0.075]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.1176</td>
      <td>[1.0, 0.0, 0.0]</td>
    </tr>
  </tbody>
</table>
</div>



Lets plot the efficient frontier considering **all** of the industries at our disposal. 

The efficient frontier shows the optimal weightings out of all possible combinations of industries we could invest in in terms of either choose a defined risk budget and get the best possible returns or choose a return goal and get the least risk. 

```python
plot_ef(n_points=50, er=er, cov=cov, style='.-', legend=True)
```




    <AxesSubplot:xlabel='Volatility'>




![png](/assets/img/2021-01-19-third_post_files/output_26_1.png)


## Capital Market Line
We will proceed with highlighting the more interesting weighting options on the efficient frontier.
Lets start with the portfolio which optimizes the return per risk. 
Instead of minimizing the volatility for given target returns, we want to maximize the SR, given a riskfree rate, to get weightings. We will present this problem slightly different so we can more easily implement it in our given framework: We will minimize the negative Sharpe Ratio.

We will also include the riskfree investment and draw a line between these two which represents different weighting schemes between this tangent portfolio and the riskfree investment. 

```python
# see also minimize_vol for similar structure
def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    Takes expected returns,covariance matrix,  riskfree rate
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1 # no shorting
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

```

Lets visualize the max SR point and the riskfree portfolio by adding additional options to our show_ef function

```python
def plot_ef(n_points, er, cov, style='.-', legend=False, riskfree_rate=0, show_cml=False):
    """
    Plots the multi-asset efficient frontier
    Takes number of target returns, exp. returns, covariance, riskfree rate
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if show_cml:
        ax.set_xlim(left = 0)
        # get MSR
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='.', linestyle='dashed', linewidth=2, markersize=12)
    return ax
```

```python
plot_ef(n_points=20, er=er, cov=cov, style='.-', legend=True, riskfree_rate = 0.02, show_cml=True)
```




    <AxesSubplot:xlabel='Volatility'>




![png](/assets/img/2021-01-19-third_post_files/output_31_1.png)


We just did some important steps! We found the efficient frontier and its tangent portfolio which optimizes the return/risk ratio. This forms the capital market line. 

We can always start with the tangent portfolio and mix in the riskfree investment at the riskfree rate. This way we walk along the capital market line and get the highest return for any defined risk. 

## Global Minimum Variance Portfolio
We will add one more possible portfolio in the grapghic. There is one portfolio which is not dependent on estimating any returns and bypasses the problems of getting extreme weightings with very small deviations in expected returns to realized returns.

The global minimum variance portfolio is not dependent on estimated returns. 

```python
def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    Takes covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)
```

Let's add the option to show the Global Minimum Variance portfolio:

```python
def plot_ef(n_points, er, cov, style='.-', legend=False, riskfree_rate=0, show_cml=False, show_gmv=False):
    """
    Plots the multi-asset efficient frontier
    Takes number of target returns, exp. returns, covariance, riskfree rate
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if show_cml:
        ax.set_xlim(left = 0)
        # get MSR
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='.', linestyle='dashed', linewidth=2, markersize=12)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        ax.plot([vol_gmv], [r_gmv], color='goldenrod', marker='.', markersize=14)
    return ax
```

```python
plot_ef(n_points=20, er=er, cov=cov, style='.-', legend=True, riskfree_rate = 0.02, show_cml=True, show_gmv=True)
```




    <AxesSubplot:xlabel='Volatility'>




![png](/assets/img/2021-01-19-third_post_files/output_37_1.png)


In this post we implemented the efficient frontier for our dataset which provides us with the weightings for the optimal return per risk. 
We leverage the covariance between our investments to remome any unsystematic risk which does not provide additional return benefits. 

We then include the capital market line as the line between the tangent portfolio with the highest return / risk ratio and the riskfree investment. An investor can purcahse this portfolio and the riskfree investment to satisfy its risk appetite and still benefit from the highest possible return per risk, outclassing the options on the efficient frontier. 

We also includes the Global Minimum Variance portfolio wich is not the optimal portfolio in term to reward per risk but has the benefit of not needing any expected returns. We have seen that the weightings significantly skew to extreme results with only small deviations from the expected returns. The GMV portfolio bypasses the problem of estimating returns (incorrectly) since it only needs the (expected) covariance as input parameter.
