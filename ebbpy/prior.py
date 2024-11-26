import numpy as np
import polars as pl
from scipy.stats import beta, betabinom
from scipy.optimize import minimize

class EbbPrior:
    """
    A class representing a beta-binomial prior estimation result.
    """

    def __init__(self, parameters, method, terms, fit=None, model=None):
        """
        Initialize the EbbPrior object.

        Parameters:
        - parameters: A dictionary with 'alpha' and 'beta' keys.
        - method: The method used for estimation ('mm', 'mle', or 'gamlss').
        - terms: A dictionary with 'x' and 'n' keys representing column names.
        - fit: The fitted model object (if applicable).
        - model: The original data used in the fit (Polars DataFrame).
        """
        self.parameters = parameters  # Contains 'alpha' and 'beta'
        self.method = method          # Estimation method ('mm', 'mle', 'gamlss')
        self.terms = terms            # Dict with 'x' and 'n' keys
        self.fit = fit                # Model fit object (if applicable)
        self.model = model            # Original data used in fitting

    def model_frame(self):
        """
        Extract the data frame with the variables needed to evaluate the empirical Bayes model fit.
        """
        return self.model

    def __str__(self):
        """
        String representation of the EbbPrior object.
        """
        params_str = self.parameters.__str__()
        return f"Empirical Bayes binomial fit with method '{self.method}'\nParameters:\n{params_str}"

def ebb_fit_prior(tbl, x, n, method='mle', mu_predictors=None, sigma_predictors=None,
                  mu_link='logit', sigma_link='log', start=None, **kwargs):
    """
    Estimate the shape parameters of a beta by fitting a beta-binomial distribution.

    Parameters:
    - tbl: Polars DataFrame
    - x: Name of the column for the number of successes
    - n: Name of the column for the total number of trials
    - method: 'mle', 'mm', or 'gamlss' (only 'mm' and 'mle' are implemented)
    - mu_predictors: Predictors for mu (ignored if method is not 'gamlss')
    - sigma_predictors: Predictors for sigma (ignored if method is not 'gamlss')
    - mu_link: Link function for mu (ignored if method is not 'gamlss')
    - sigma_link: Link function for sigma (ignored if method is not 'gamlss')
    - start: Starting values for 'mle' method (dictionary with 'alpha' and 'beta')
    - kwargs: Additional arguments passed to the optimizer (if method is 'mle')

    Returns:
    - An EbbPrior object containing the estimated parameters
    """
    return _ebb_fit_prior(tbl, x, n, method=method, mu_predictors=mu_predictors,
                          sigma_predictors=sigma_predictors, mu_link=mu_link,
                          sigma_link=sigma_link, start=start, **kwargs)

def _ebb_fit_prior(tbl, x, n, method='mle', mu_predictors=None, sigma_predictors=None,
                   mu_link='logit', sigma_link='log', start=None, **kwargs):
    """
    Internal function to estimate the shape parameters of a beta by fitting a beta-binomial distribution.
    """
    x_value = tbl[x].to_numpy()
    n_value = tbl[n].to_numpy()

    # Check if there are predictors
    no_predictors = (mu_predictors is None or mu_predictors == '~1') and \
                    (sigma_predictors is None or sigma_predictors == '~1')
    if not no_predictors:
        method = 'gamlss'

    if method == 'mm':
        # Method of moments
        p = x_value / n_value
        mu = np.mean(p)
        vr = np.var(p, ddof=1)  # Sample variance

        # Handle cases where variance is zero or extremely small
        if vr <= 0 or np.isclose(vr, 0):
            raise ValueError("Variance is zero or too small for method of moments estimation.")

        # Compute alpha and beta
        alpha = ((1 - mu) / vr - 1 / mu) * mu ** 2
        beta_param = alpha * (1 / mu - 1)

        # Ensure alpha and beta are positive
        if alpha <= 0 or beta_param <= 0:
            raise ValueError("Method of moments estimation resulted in non-positive alpha or beta.")

        parameters = {'alpha': alpha, 'beta': beta_param}
        fit = None
    elif method == 'mle':
        # Maximum likelihood estimation
        # Get initial estimates from method of moments if start is None
        if start is None:
            mm_estimate = _ebb_fit_prior(tbl, x, n, method='mm')
            start_alpha = mm_estimate.parameters['alpha']
            start_beta = mm_estimate.parameters['beta']
        else:
            start_alpha = start['alpha']
            start_beta = start['beta']

        # Define negative log-likelihood function
        def neg_log_likelihood(params):
            alpha, beta_param = params
            # Avoid invalid values
            if alpha <= 0 or beta_param <= 0:
                return np.inf
            # Compute negative log-likelihood
            log_likelihood = betabinom.logpmf(x_value, n_value, alpha, beta_param)
            return -np.sum(log_likelihood)

        # Minimize negative log-likelihood
        result = minimize(
            neg_log_likelihood,
            x0=[start_alpha, start_beta],
            bounds=[(1e-9, None), (1e-9, None)],
            method='L-BFGS-B',
            **kwargs
        )

        if not result.success:
            raise RuntimeError('MLE optimization failed: ' + result.message)

        alpha, beta_param = result.x
        parameters = {'alpha': alpha, 'beta': beta_param}
        fit = result
    elif method == 'gamlss':
        # Beta-binomial regression (not implemented)
        raise NotImplementedError("The 'gamlss' method is not implemented in this translation.")
    else:
        raise ValueError("Invalid method specified. Choose 'mle', 'mm', or 'gamlss'.")

    # Prepare terms and model
    terms = {'x': x, 'n': n, 'mu_predictors': mu_predictors, 'sigma_predictors': sigma_predictors}
    vars_used = [x, n]
    if mu_predictors is not None:
        vars_used.extend(mu_predictors)
    if sigma_predictors is not None:
        vars_used.extend(sigma_predictors)
    vars_used = list(set(vars_used))
    model = tbl.select(vars_used)

    # Create EbbPrior object
    ebb_prior = EbbPrior(parameters=parameters, fit=fit, terms=terms, method=method, model=model)

    return ebb_prior

def augment(ebb_prior, data=None, cred_level=0.95, newdata=None):
    """
    Augment the data with additional columns from the beta-binomial estimation.

    Parameters:
    - ebb_prior: An instance of EbbPrior class.
    - data: Polars DataFrame containing the original data (if None, uses ebb_prior.model).
    - cred_level: The level of credible intervals for each observation.
                  If None, credible intervals are not computed.
    - newdata: Polars DataFrame with new data to augment (overrides 'data' if provided).

    Returns:
    - Polars DataFrame with additional columns:
        - '.alpha1': Posterior alpha parameter for each observation.
        - '.beta1': Posterior beta parameter for each observation.
        - '.fitted': Posterior mean (shrunk estimate).
        - '.raw': Raw estimate (successes / trials).
        - '.low': Lower bound of credible interval (if cred_level is specified).
        - '.high': Upper bound of credible interval (if cred_level is specified).
    """
    if data is None:
        data = ebb_prior.model_frame()
    if newdata is not None:
        data = newdata

    x_col = ebb_prior.terms['x']
    n_col = ebb_prior.terms['n']

    if ebb_prior.method == 'gamlss':
        # For 'gamlss', predict 'mu' and 'sigma' using the fit object
        raise NotImplementedError("The 'gamlss' method is not implemented.")
        # Implementations for 'gamlss' would go here
    else:
        alpha = ebb_prior.parameters['alpha']
        beta_param = ebb_prior.parameters['beta']
        data = data.with_columns([
            (pl.col(x_col) + alpha).alias('.alpha1'),
            (pl.col(n_col) - pl.col(x_col) + beta_param).alias('.beta1')
        ])

    # Compute '.fitted' and '.raw' columns
    data = data.with_columns([
        (pl.col('.alpha1') / (pl.col('.alpha1') + pl.col('.beta1'))).alias('.fitted'),
        (pl.col(x_col) / pl.col(n_col)).alias('.raw')
    ])

    if cred_level is not None:
        lower_quantile = (1 - cred_level) / 2
        upper_quantile = 1 - lower_quantile

        # Extract '.alpha1' and '.beta1' as NumPy arrays for vectorized computation
        alpha1 = data['.alpha1'].to_numpy()
        beta1 = data['.beta1'].to_numpy()

        # Compute credible intervals
        low = beta.ppf(lower_quantile, alpha1, beta1)
        high = beta.ppf(upper_quantile, alpha1, beta1)

        # Add '.low' and '.high' columns to the DataFrame
        data = data.with_columns([
            pl.Series('.low', low),
            pl.Series('.high', high)
        ])

    return data

def tidy(ebb_prior):
    """
    Tidy the results of beta-binomial estimation.

    Parameters:
    - ebb_prior: An instance of EbbPrior class.

    Returns:
    - A Polars DataFrame containing the parameters.
    """
    # Convert parameters to Polars DataFrame if necessary
    if isinstance(ebb_prior.parameters, dict):
        ret = pl.DataFrame([ebb_prior.parameters])
    else:
        ret = ebb_prior.parameters

    if ebb_prior.method != 'gamlss':
        # Add 'mean' column
        ret = ret.with_columns(
            (pl.col('alpha') / (pl.col('alpha') + pl.col('beta'))).alias('mean')
        )
    return ret

def glance(ebb_prior):
    """
    Glance at the summary of the ebb_prior object.

    Parameters:
    - ebb_prior: An instance of EbbPrior class.

    Returns:
    - A Polars DataFrame or dictionary with summary statistics.
    """
    if ebb_prior.method == 'gamlss':
        # Return the degrees of freedom (number of parameters)
        ret = pl.DataFrame({'df': [len(tidy(ebb_prior).columns)]})
    else:
        # Return 'alpha', 'beta', and 'mean'
        tdy = tidy(ebb_prior)
        ret = tdy.select(['alpha', 'beta', 'mean'])
    return ret

def add_ebb_estimate(tbl, x, n, cred_level=0.95, prior_subset=None, **kwargs):
    """
    Perform empirical Bayes shrinkage on a DataFrame.

    Parameters:
    - tbl: Polars DataFrame.
    - x: Column name containing number of successes.
    - n: Column name containing totals.
    - cred_level: Level of credible interval to compute. If None, do not compute intervals.
    - prior_subset: A boolean mask or callable that returns a boolean mask indicating which rows should be used for computing the prior.
    - kwargs: Additional arguments passed on to ebb_fit_prior, such as method.

    Returns:
    - The original table, with several columns added based on empirical Bayes shrinkage:
        - '.alpha1': Posterior alpha (shape1) parameter
        - '.beta1': Posterior beta (shape2) parameter
        - '.fitted': Posterior shrunken estimate
        - '.raw': Estimate without shrinkage (success / total)
        - '.low': Lower bound of credible interval
        - '.high': Upper bound of credible interval
    """
    # Evaluate prior_subset
    if prior_subset is None:
        prior_tbl = tbl
    else:
        if callable(prior_subset):
            mask = prior_subset(tbl)
            prior_tbl = tbl.filter(mask)
        else:
            # Assume prior_subset is a boolean Series or array
            prior_tbl = tbl.filter(pl.Series(prior_subset))

    # Estimate prior
    prior = ebb_fit_prior(prior_tbl, x, n, **kwargs)

    # Use the prior to augment the original table
    augmented_tbl = augment(ebb_prior=prior, data=tbl, cred_level=cred_level, newdata=tbl)

    return augmented_tbl
