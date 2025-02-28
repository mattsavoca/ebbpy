# ebbpy: Empirical Bayes on the Binomial in Python with Polars

**Acknowledgments**

The `ebbpy` package in Python is inspired by the [ebbr](https://github.com/dgrtwo/ebbr) package in R, developed by [David Robinson](https://github.com/dgrtwo). The methods and examples are based on his work on empirical Bayes estimation and the accompanying blog posts.

**Contributing**

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on GitHub.
**License:** MIT

---

**Overview**

The `ebbpy` package provides methods for empirical Bayes shrinkage and estimation on data with many observations of success/total counts. These methods are useful for adjusting estimates towards a global prior, especially when dealing with proportions estimated from varying sample sizes. The ebbpy package in Python is inspired by the [`ebbr`](https://github.com/dgrtwo/ebbr) package in R, developed by David Robinson. The methods and examples are based on his work on empirical Bayes estimation and the accompanying blog posts.


**Installation**

Currently, `ebbpy` is not available on PyPI. You can install the package from GitHub:

```bash
pip install git+https://github.com/mattsavoca/ebbpy.git
```

Alternatively, you can clone the repository and install it locally:

```bash
git clone https://github.com/mattsavoca/ebbpy.git
cd ebbpy
pip install .
```

**Functions**

`ebbpy` provides several functions for empirical Bayes estimation:

- `ebb_fit_prior`: Fits a beta distribution to a dataset of success/total counts using maximum likelihood estimation or the method of moments. It returns an `EbbPrior` object, which includes methods like `model_frame` to retrieve the data used and `augment` to update observations with the prior.

- `add_ebb_estimate`: A shortcut for performing `ebb_fit_prior` to fit a prior, then updating each observation to create a posterior. It adds columns to the DataFrame with the posterior estimates.

- `add_ebb_prop_test`: Performs an empirical Bayesian version of a one-sample or two-sample proportion test, comparing each observation in the data to either a fixed threshold or to another beta posterior.

- `ebb_fit_mixture`: Fits a mixture of beta distributions as the prior.
---

**References**

- [Empirical Bayes Estimation](https://en.wikipedia.org/wiki/Empirical_Bayes_method)
- [David Robinson's Blog Posts on Empirical Bayes](https://varianceexplained.org/r/empirical_bayes_baseball/)
