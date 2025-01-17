# ebbpy: Empirical Bayes on the Binomial in Python

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

**Example**

Suppose we analyze the success rate of quarterbacks based on their drive success in NFL games. We can use the `calculate_tdsr` function to compute the True Drive Success Rate (TDSR) for each quarterback:

```python
import polars as pl
from ebbpy import calculate_tdsr

# Load your play-by-play data
pbp_raw = load_local_pbp(seasons=[2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
pbp = generate_primary_passers(pbp_raw)

# Calculate TDSR
tdsr_df = calculate_tdsr(pbp_df=pbp, min_plays=15, scale_metrics=False)

# Display the results
print(tdsr_df)
```

Sample Output:

```
 qb  ┆ epa ┆ cpo ┆ dri ┆ dri ┆ tds ┆ pla ┆ tea │ 
│ --- ┆ --- ┆ e   ┆ ve_ ┆ ves ┆ r   ┆ yer ┆ m   │ 
│ str ┆ f32 ┆ --- ┆ suc ┆ --- ┆ --- ┆ --- ┆ --- │ 
│     ┆     ┆ f32 ┆ ces ┆ u32 ┆ f64 ┆ str ┆ str │ 
│     ┆     ┆     ┆ s   ┆     ┆     ┆     ┆     │ 
│     ┆     ┆     ┆ --- ┆     ┆     ┆     ┆     │ 
│     ┆     ┆     ┆ i32 ┆     ┆     ┆     ┆     │ 
╞═════╪═════╪═════╪═════╪═════╪═════╪═════╪═════╡ 
│ 00- ┆ 1.0 ┆ -9. ┆ 59  ┆ 146 ┆ 0.3 ┆ Jos ┆ BUF │ 
│ 003 ┆ 005 ┆ 100 ┆     ┆     ┆ 552 ┆ h   ┆     │ 
│ 485 ┆ 11  ┆ 121 ┆     ┆     ┆ 01  ┆ All ┆     │ 
│ 7   ┆     ┆     ┆     ┆     ┆     ┆ en  ┆     │ 
│ 00- ┆ 0.9 ┆ -4. ┆ 65  ┆ 164 ┆ 0.3 ┆ Jar ┆ DET │ 
│ 003 ┆ 540 ┆ 459 ┆     ┆     ┆ 533 ┆ ed  ┆     │ 
│ 310 ┆ 23  ┆ 463 ┆     ┆     ┆ 81  ┆ Gof ┆     │ 
│ 6   ┆     ┆     ┆     ┆     ┆     ┆ f   ┆     │ 
│ 00- ┆ 1.1 ┆ -6. ┆ 61  ┆ 160 ┆ 0.3 ┆ Lam ┆ BAL │ 
│ 003 ┆ 107 ┆ 494 ┆     ┆     ┆ 412 ┆ ar  ┆     │ 
│ 479 ┆ 13  ┆ 303 ┆     ┆     ┆ 02  ┆ Jac ┆     │ 
│ 6   ┆     ┆     ┆     ┆     ┆     ┆ kso ┆     │ 
│     ┆     ┆     ┆     ┆     ┆     ┆ n   ┆     │ 
│ 00- ┆ 0.6 ┆ -8. ┆ 58  ┆ 174 ┆ 0.3 ┆ Bak ┆ TB  │ 
│ 003 ┆ 923 ┆ 553 ┆     ┆     ┆ 068 ┆ er  ┆     │ 
│ 485 ┆ 49  ┆ 976 ┆     ┆     ┆ 46  ┆ May ┆     │ 
│ 5   ┆     ┆     ┆     ┆     ┆     ┆ fie ┆     │ 
│     ┆     ┆     ┆     ┆     ┆     ┆ ld  ┆     │ 
│ 00- ┆ 0.6 ┆ -3. ┆ 54  ┆ 166 ┆ 0.2 ┆ Joe ┆ CIN │ 
│ 003 ┆ 008 ┆ 022 ┆     ┆     ┆ 997 ┆ Bur ┆     │ 
│ 644 ┆ 41  ┆ 271 ┆     ┆     ┆ 69  ┆ row ┆     │ 
│ 2   ┆     ┆     ┆     ┆     ┆     ┆     ┆     │ 
│ …   ┆ …   ┆ …   ┆ …   ┆ …   ┆ …   ┆ …   ┆ …   │ 
│ 00- ┆ -0. ┆ -7. ┆ 14  ┆ 93  ┆ 0.1 ┆ Gar ┆ LV  │ 
│ 003 ┆ 494 ┆ 761 ┆     ┆     ┆ 751 ┆ dne ┆     │ 
│ 528 ┆ 739 ┆ 265 ┆     ┆     ┆ 36  ┆ r   ┆     │ 
│ 9   ┆     ┆     ┆     ┆     ┆     ┆ Min ┆     │ 
│     ┆     ┆     ┆     ┆     ┆     ┆ she ┆     │ 
│     ┆     ┆     ┆     ┆     ┆     ┆ w   ┆     │ 
│ 00- ┆ -0. ┆ -10 ┆ 7   ┆ 56  ┆ 0.1 ┆ Jac ┆ NE  │ 
│ 003 ┆ 662 ┆ .11 ┆     ┆     ┆ 703 ┆ oby ┆     │ 
│ 311 ┆ 762 ┆ 525 ┆     ┆     ┆ 37  ┆ Bri ┆     │ 
│ 9   ┆     ┆ 2   ┆     ┆     ┆     ┆ sse ┆     │ 
│     ┆     ┆     ┆     ┆     ┆     ┆ tt  ┆     │
│ 00- ┆ -2. ┆ -23 ┆ 2   ┆ 32  ┆ 0.1 ┆ Dor ┆ CLE │
│ 003 ┆ 223 ┆ .42 ┆     ┆     ┆ 595 ┆ ian ┆     │
│ 858 ┆ 161 ┆ 382 ┆     ┆     ┆ 27  ┆ Tho ┆     │
│ 3   ┆     ┆ 6   ┆     ┆     ┆     ┆ mps ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆ on- ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆ Rob ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆ ins ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆ on  ┆     │
│ 00- ┆ -0. ┆ -12 ┆ 8   ┆ 71  ┆ 0.1 ┆ Des ┆ CLE │
│ 003 ┆ 866 ┆ .17 ┆     ┆     ┆ 577 ┆ hau ┆     │
│ 353 ┆ 778 ┆ 096 ┆     ┆     ┆ 31  ┆ n   ┆     │
│ 7   ┆     ┆ 2   ┆     ┆     ┆     ┆ Wat ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆ son ┆     │
│ 00- ┆ -1. ┆ -17 ┆ 7   ┆ 67  ┆ 0.1 ┆ Spe ┆ NO  │
│ 003 ┆ 265 ┆ .29 ┆     ┆     ┆ 546 ┆ nce ┆     │
│ 937 ┆ 68  ┆ 677 ┆     ┆     ┆ 39  ┆ r   ┆     │
│ 6   ┆     ┆ 8   ┆     ┆     ┆     ┆ Rat ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆ tle ┆     │
│     ┆     ┆     ┆     ┆     ┆     ┆ r   ┆     
```

This output shows the TDSR, expected points added (EPA), completion percentage over expected (CPOE), total drives, and successful drives for each quarterback.

The `add_ebb_estimate` function can then be used to apply empirical Bayes shrinkage to the TDSR estimates:

```python
from ebbpy import add_ebb_estimate

# Perform empirical Bayes shrinkage on TDSR
shrunken_tdsr = add_ebb_estimate(tdsr_df, x='drive_success', n='drives')

# Display the augmented DataFrame
print(shrunken_tdsr.select(['qb', 'tdsr', '.fitted', '.raw']).head())
```

This adds columns to the data, including the raw TDSR estimate (`.raw`) and the shrunken empirical Bayes estimate (`.fitted`).

**Code of Conduct**

Please note that this project is released with a [Contributor Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/). By participating in this project, you agree to abide by its terms.

---

**Dependencies**

Ensure that you have the necessary packages installed:

```bash
pip install polars numpy scipy matplotlib
```

**Note**

- **Variable Naming:** In the code examples, ensure that the column names in your DataFrame match those used in the functions (e.g., `'drive_success'` for successes and `'drives'` for trials).

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
