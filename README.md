# ebbpy: Empirical Bayes on the Binomial in Python

The `ebbpy` package is python / polars refactoring of the `{ebbr}` R package by David Robinson.

<!---
**License:** MIT

---

**Overview**

The `ebbpy` package provides methods for empirical Bayes shrinkage and estimation on data with many observations of success/total counts. These methods are useful for adjusting estimates towards a global prior, especially when dealing with proportions estimated from varying sample sizes. The methods are inspired by a series of blog posts on baseball batting averages but can be applied to a variety of data types.

**Installation**

Currently, `ebbpy` is not available on PyPI. You can install the package from GitHub:

```bash
pip install git+https://github.com/yourusername/ebbpy.git
```

Alternatively, you can clone the repository and install it locally:

```bash
git clone https://github.com/yourusername/ebbpy.git
cd ebbpy
pip install .
```

**Functions**

`ebbpy` provides several functions for empirical Bayes estimation:

- `ebb_fit_prior`: Fits a beta distribution to a dataset of success/total counts using maximum likelihood estimation or the method of moments. It returns an `EbbPrior` object, which includes methods like `model_frame` to retrieve the data used and `augment` to update observations with the prior.

- `add_ebb_estimate`: A shortcut for performing `ebb_fit_prior` to fit a prior, then updating each observation to create a posterior. It adds columns to the DataFrame with the posterior estimates.

- `add_ebb_prop_test`: Performs an empirical Bayesian version of a one-sample or two-sample proportion test, comparing each observation in the data to either a fixed threshold or to another beta posterior.

- `ebb_fit_mixture`: Fits a mixture of beta distributions as the prior.

**Example**

Suppose we simulate some data from a beta-binomial model. Each observation has a true probability drawn from a beta distribution (with parameters `alpha=10` and `beta=40`, resulting in a mean of 20%). However, the totals vary, such that our estimate of `x / total` has a lot of noise for some observations:

```python
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# Simulate data
np.random.seed(2017)
obs = 1000
prob = np.random.beta(10, 40, size=obs)
total = np.round(np.random.lognormal(mean=4, sigma=2, size=obs)) + 1
total = total.astype(int)
x = np.random.binomial(total, prob)
sim_dat = pl.DataFrame({'prob': prob, 'total': total, 'x': x})

# Plot the data
plt.scatter(sim_dat['total'], sim_dat['x'] / sim_dat['total'])
plt.xscale('log')
plt.xlabel('total')
plt.ylabel('x / total')
plt.title('Simulated Data')
plt.show()
```

![Simulated Data Plot](path_to_plot_image)

We would want to shrink towards a beta prior, a process described [here](https://varianceexplained.org/r/empirical_bayes_baseball/). We can fit a beta prior using `ebb_fit_prior`:

```python
# Import the function
from ebbpy import ebb_fit_prior

# Fit the prior
prior = ebb_fit_prior(sim_dat, x='x', n='total', method='mle')

# Display the prior parameters
print("Prior Parameters:")
print(prior.parameters)
```

Sample Output:

```
Prior Parameters:
{'alpha': 11.39114826827856, 'beta': 46.08749800382458}
```

Notice that the function takes the DataFrame first, since it is designed to be chainable, and the following two arguments are the success column (`x`) and the total column (`n`). This computes an estimate of the prior based on maximum likelihood estimation and gets rather close to the true alpha and beta values.

We can then use that prior to update each individual observation. The `add_ebb_estimate` function is a shortcut for both fitting the prior and updating observations, which is the most common use case:

```python
# Import the function
from ebbpy import add_ebb_estimate

# Perform empirical Bayes shrinkage
shrunken = add_ebb_estimate(sim_dat, x='x', n='total')

# Display the augmented DataFrame
print(shrunken.select(['prob', 'total', 'x', '.alpha1', '.beta1', '.fitted', '.raw']).head())
```

Sample Output:

```
shape: (5, 7)
┌───────────┬───────┬─────┬──────────┬──────────┬──────────┬──────────┐
│ prob      ┆ total ┆ x   ┆ .alpha1  ┆ .beta1   ┆ .fitted  ┆ .raw     │
│ ---       ┆ ---   ┆ --- ┆ ---      ┆ ---      ┆ ---      ┆ ---      │
│ f64       ┆ i64   ┆ i64 ┆ f64      ┆ f64      ┆ f64      ┆ f64      │
╞═══════════╪═══════╪═════╪══════════╪══════════╪══════════╪══════════╡
│ 0.320065  ┆ 182   ┆ 66  ┆ 77.39115 ┆ 162.0875 ┆ 0.323165 ┆ 0.362637 │
│ 0.195053  ┆ 1015  ┆ 198 ┆ 209.3911 ┆ 863.0875 ┆ 0.195240 ┆ 0.195074 │
│ 0.253414  ┆ 5     ┆ 1   ┆ 12.39115 ┆ 50.0875  ┆ 0.198326 ┆ 0.200000 │
│ 0.100220  ┆ 12    ┆ 3   ┆ 14.39115 ┆ 55.0875  ┆ 0.207131 ┆ 0.250000 │
│ 0.195527  ┆ 1259  ┆ 223 ┆ 234.3911 ┆ 1082.087 ┆ 0.178044 ┆ 0.177125 │
└───────────┴───────┴─────┴──────────┴──────────┴──────────┴──────────┘
```

This adds columns to the data, including the raw `x / total` estimate (`.raw`) and the shrunken empirical Bayes estimate (`.fitted`):

```python
# Visualize the shrinkage towards the prior mean
plt.scatter(shrunken['.raw'], shrunken['.fitted'], c=np.log10(shrunken['total']), cmap='viridis')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
prior_mean = prior.parameters['alpha'] / (prior.parameters['alpha'] + prior.parameters['beta'])
plt.axhline(y=prior_mean, color='red', linestyle=':', label=f"Prior Mean ({prior_mean:.2f})")
plt.xlabel('.raw (Observed Proportion)')
plt.ylabel('.fitted (Posterior Estimate)')
plt.colorbar(label='log10(total)')
plt.title('Empirical Bayes Shrinkage')
plt.legend()
plt.show()
```

![Shrinkage Plot](path_to_shrinkage_plot_image)

The output shows how observations with smaller total counts are shrunk more towards the prior mean, while those with larger counts remain closer to their observed proportions.

The output also includes credible intervals for each observation. For example, we can examine the estimates and credible intervals of the first 20 observations, and compare them to the true probabilities:

```python
# Add observation index for plotting
first_20 = shrunken.head(20).with_row_count(name='observation')

# Plotting the estimates and credible intervals for the first 20 observations
plt.errorbar(first_20['observation'], first_20['.fitted'], yerr=[first_20['.fitted'] - first_20['.low'], first_20['.high'] - first_20['.fitted']], fmt='o', label='Empirical Bayes Estimate')
plt.scatter(first_20['observation'], first_20['prob'], color='red', label='True Probability')
plt.xlabel('Observation')
plt.ylabel('Estimate')
plt.title('Estimating Proportions in 20 Observations')
plt.legend()
plt.show()
```

![Credible Intervals Plot](path_to_credible_intervals_plot_image)

As expected, the 95% credible intervals contain the true probabilities about 95% of the time.

**Code of Conduct**

Please note that this project is released with a [Contributor Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/). By participating in this project, you agree to abide by its terms.

---

**Dependencies**

Ensure that you have the necessary packages installed:

```bash
pip install polars numpy scipy matplotlib
```

**Note**

- **Variable Naming:** In the code examples, ensure that the column names in your DataFrame match those used in the functions (e.g., `'x'` for successes and `'total'` for trials).

- **Data Handling:** The code uses [Polars](https://pola.rs/) for efficient data manipulation. Polars provides a DataFrame API similar to pandas but with improved performance.

- **Visualization:** The plots are created using `matplotlib`. You can customize the plots or use other visualization libraries like `seaborn` if preferred.

---

**Acknowledgments**

The `ebbpy` package in Python is inspired by the [ebbr](https://github.com/dgrtwo/ebbr) package in R, developed by [David Robinson](https://github.com/dgrtwo). The methods and examples are based on his work on empirical Bayes estimation and the accompanying blog posts.

**Contributing**

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on GitHub.

---

**License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**References**

- [Empirical Bayes Estimation](https://en.wikipedia.org/wiki/Empirical_Bayes_method)
- [David Robinson's Blog Posts on Empirical Bayes](https://varianceexplained.org/r/empirical_bayes_baseball/)
