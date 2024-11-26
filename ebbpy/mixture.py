# mixture.py

import numpy as np
import polars as pl
from scipy.stats import betabinom
from scipy.optimize import minimize
from .prior import ebb_fit_prior  # Importing ebb_fit_prior from prior.py


class EbbMixture:
    """
    A class representing a mixture model of beta-binomial distributions.
    """

    def __init__(self, clusters, assignments, num_iter, model=None, vary_size=False):
        """
        Initialize the EbbMixture object.

        Parameters:
        - clusters: A Polars DataFrame containing hyperparameters for each cluster.
        - assignments: A Polars DataFrame with '.cluster' and '.likelihood' for each observation.
        - num_iter: The number of iterations performed during fitting.
        - model: The original data the mixture model was fit on (optional).
        - vary_size: Boolean indicating whether cluster sizes vary (default False).
        """
        self.clusters = clusters          # DataFrame with cluster hyperparameters
        self.assignments = assignments    # DataFrame with cluster assignments and likelihoods
        self.num_iter = num_iter          # Number of iterations in the fitting process
        self.model = model                # Original data (optional)
        self.vary_size = vary_size        # Indicates if cluster sizes vary

    def tidy(self):
        """
        Tidy a mixture of beta-binomials.

        Returns:
        - A Polars DataFrame with one row for each cluster, containing:
            - 'cluster': Cluster IDs
            - 'alpha': Alpha shape parameter for the cluster
            - 'beta': Beta shape parameter for the cluster
            - 'mean': Mean of the beta distribution (alpha / (alpha + beta))
            - 'size': Size of the cluster (number of observations assigned)
            - 'probability': Prior probability of being in this cluster
        """
        clusters_df = self.clusters

        # Compute 'mean' if not already present
        if 'mean' not in clusters_df.columns:
            clusters_df = clusters_df.with_columns(
                (pl.col('alpha') / (pl.col('alpha') + pl.col('beta'))).alias('mean')
            )

        # Compute 'size' based on assignments if not provided
        if 'size' not in clusters_df.columns:
            cluster_sizes = (
                self.assignments
                .groupby('.cluster')
                .agg(pl.count().alias('size'))
            )
            clusters_df = clusters_df.join(cluster_sizes, on='cluster', how='left')

        return clusters_df

    def augment(self, data):
        """
        Augment the data with cluster assignments and likelihoods.

        Parameters:
        - data: Polars DataFrame containing the original data.

        Returns:
        - Polars DataFrame with additional columns:
            - '.cluster': Cluster assignments for each observation
            - '.likelihood': Likelihood of each observation under the assigned cluster
        """
        assignments_df = self.assignments

        # Ensure that the assignments have the same number of rows as data
        if len(assignments_df) != len(data):
            raise ValueError("Assignments and data must have the same number of rows.")

        # Add '.cluster' and '.likelihood' columns to the data
        augmented_data = data.with_columns([
            assignments_df['.cluster'],
            assignments_df['.likelihood']
        ])

        return augmented_data

    def glance(self):
        """
        Glance at the mixture model.

        Returns:
        - A Polars DataFrame with summary statistics:
            - 'iter': Number of iterations
            - 'logLik': Log-likelihood of the model given the final assignments
            - 'AIC': Akaike Information Criterion
            - 'BIC': Bayesian Information Criterion
        """
        # Compute log-likelihood
        log_likelihood_info = self.log_likelihood()
        logLik = log_likelihood_info['loglik']
        df = log_likelihood_info['df']
        nobs = log_likelihood_info['nobs']

        # Compute AIC and BIC
        AIC = 2 * df - 2 * logLik
        BIC = np.log(nobs) * df - 2 * logLik

        # Create a summary DataFrame
        glance_df = pl.DataFrame({
            'iter': [self.num_iter],
            'logLik': [logLik],
            'AIC': [AIC],
            'BIC': [BIC]
        })

        return glance_df

    def log_likelihood(self):
        """
        Compute the log-likelihood of the EbbMixture object.

        Returns:
        - A dictionary containing:
            - 'loglik': Log-likelihood value
            - 'df': Degrees of freedom
            - 'nobs': Number of observations
        """
        # Ensure that the original data is available
        if self.model is None:
            raise ValueError("Original data must be provided in the EbbMixture object to compute log-likelihood.")

        data = self.model
        assignments = self.assignments
        clusters = self.clusters

        # Merge data with assignments
        data_with_assignments = data.with_columns([
            assignments['.cluster'],
            assignments['.likelihood']
        ])

        # Merge with cluster parameters
        clusters = clusters.rename({'cluster': '.cluster'})
        data_with_clusters = data_with_assignments.join(
            clusters, on='.cluster', how='left'
        )

        # Extract necessary columns
        x_col = self.model.columns[0]  # Assuming the first column is 'x'
        n_col = self.model.columns[1]  # Assuming the second column is 'n'

        x = data_with_clusters[x_col].to_numpy()
        n = data_with_clusters[n_col].to_numpy()
        alpha = data_with_clusters['alpha'].to_numpy()
        beta_param = data_with_clusters['beta'].to_numpy()

        # Compute log-likelihood using beta-binomial distribution
        log_pmf = betabinom.logpmf(x, n, alpha, beta_param)
        logLik = np.sum(log_pmf)

        # Degrees of freedom
        df = 2 * len(clusters)
        if self.vary_size:
            df += len(clusters) - 1

        nobs = len(assignments)

        # Return a dictionary containing the log-likelihood, degrees of freedom, and number of observations
        return {'loglik': logLik, 'df': df, 'nobs': nobs}

    def __str__(self):
        """
        String representation of the EbbMixture object.

        Returns:
        - A string representation of the EbbMixture object.
        """
        num_clusters = len(self.clusters)
        cluster_info = self.tidy().to_pandas().to_string(index=False)
        return f"A mixture of {num_clusters} beta-binomial distributions\n\nCluster parameters:\n{cluster_info}"


def estimate_clusters(assignments, x_col, n_col, method, **kwargs):
    """
    Estimate parameters for each cluster.

    Parameters:
    - assignments: Polars DataFrame with current cluster assignments.
    - x_col: Name of the x column.
    - n_col: Name of the n column.
    - method: Method to pass to ebb_fit_prior ('mle' or 'mm').

    Returns:
    - Polars DataFrame with parameters for each cluster.
    """
    clusters_list = []
    unique_clusters = assignments['.cluster'].unique().to_list()
    for cluster_label in unique_clusters:
        cluster_data = assignments.filter(pl.col('.cluster') == cluster_label)
        cluster_tbl = cluster_data.select([x_col, n_col])
        eb_prior = ebb_fit_prior(cluster_tbl, x_col, n_col, method=method, **kwargs)
        alpha = eb_prior.parameters['alpha']
        beta_param = eb_prior.parameters['beta']
        mean = alpha / (alpha + beta_param)
        number = cluster_data.shape[0]
        clusters_list.append({
            'cluster': cluster_label,
            'alpha': alpha,
            'beta': beta_param,
            'mean': mean,
            'number': number
        })
    fits = pl.DataFrame(clusters_list)
    return fits


def update_assignments(assignments, fits, x_col, n_col):
    """
    Update the cluster assignments based on the computed parameters.

    Parameters:
    - assignments: Polars DataFrame with current assignments.
    - fits: Polars DataFrame with cluster parameters.
    - x_col: Name of the x column.
    - n_col: Name of the n column.

    Returns:
    - Polars DataFrame with updated assignments.
    """
    data = assignments.select(['id', x_col, n_col])

    data = data.with_column(pl.lit(1).alias('key'))
    fits = fits.with_column(pl.lit(1).alias('key'))

    cross_joined = data.join(fits, on='key', how='inner').drop('key')

    x = cross_joined[x_col].to_numpy()
    n = cross_joined[n_col].to_numpy()
    alpha = cross_joined['alpha'].to_numpy()
    beta_param = cross_joined['beta'].to_numpy()
    probability = cross_joined.get('probability', pl.Series('probability', np.ones(len(cross_joined)) / len(fits))).to_numpy()

    likelihoods = probability * betabinom.pmf(x, n, alpha, beta_param)
    cross_joined = cross_joined.with_column(pl.Series('.likelihood', likelihoods))

    # Select the cluster with the highest likelihood for each observation
    cross_joined = cross_joined.sort(by=['id', '.likelihood'], reverse=[False, True])
    unique_assignments = cross_joined.unique(subset=['id'], keep='first')

    updated_assignments = unique_assignments.select(['id', x_col, n_col, 'cluster', '.likelihood'])
    updated_assignments = updated_assignments.rename({'cluster': '.cluster'})
    updated_assignments = updated_assignments.sort('id')

    return updated_assignments


def ebb_fit_mixture(tbl, x, n, clusters=2, iter_max=10, nstart=1, vary_size=False, method='mle', **kwargs):
    """
    Fit a mixture of beta-binomial distributions.

    Parameters:
    - tbl: Polars DataFrame.
    - x: Name of the column for the number of successes.
    - n: Name of the column for the total number of trials.
    - clusters: Number of clusters (default 2).
    - iter_max: Maximum number of iterations to perform.
    - nstart: Number of random restarts.
    - vary_size: Allow each cluster to have a prior probability.
    - method: Method passed on to ebb_fit_prior ('mle' or 'mm').
    - kwargs: Additional arguments passed on to ebb_fit_prior.

    Returns:
    - An EbbMixture object containing the fitted mixture model.
    """
    if method == 'gamlss':
        raise NotImplementedError("Fitting a mixture of 'gamlss' models is not implemented.")

    if nstart > 1:
        mixtures = [ _ebb_fit_mixture(tbl, x, n, clusters, iter_max, vary_size, method, **kwargs)
                     for _ in range(nstart) ]
        log_liks = [ mixture.log_likelihood()['loglik'] for mixture in mixtures ]
        best_mixture = mixtures[np.argmax(log_liks)]
        return best_mixture
    else:
        return _ebb_fit_mixture(tbl, x, n, clusters, iter_max, vary_size, method, **kwargs)


def _ebb_fit_mixture(tbl, x, n, clusters=2, iter_max=10, vary_size=False, method='mle', **kwargs):
    """
    Internal function to fit the mixture model.
    """
    x_values = tbl[x].to_numpy()
    n_values = tbl[n].to_numpy()
    num_samples = len(tbl)

    cluster_ids = np.arange(1, clusters + 1)
    initial_clusters = np.random.choice(cluster_ids, size=num_samples, replace=True)

    assignments = pl.DataFrame({
        'id': np.arange(num_samples),
        x: x_values,
        n: n_values,
        '.cluster': initial_clusters.astype(str)
    })

    previous_assignments = None

    for iteration in range(iter_max):
        fits = estimate_clusters(assignments, x, n, method, **kwargs)

        if vary_size:
            total_size = fits['number'].sum()
            fits = fits.with_columns(
                (pl.col('number') / total_size).alias('probability')
            )
        else:
            fits = fits.with_columns(
                pl.lit(1 / clusters).alias('probability')
            )

        assignments = update_assignments(assignments, fits, x, n)

        # Check for convergence
        if previous_assignments is not None:
            if (assignments['.cluster'] == previous_assignments['.cluster']).all():
                break

        previous_assignments = assignments

    else:
        print("Warning: Expectation-maximization algorithm did not converge within the maximum number of iterations.")

    ebb_mixture = EbbMixture(
        clusters=fits,
        assignments=assignments,
        num_iter=iteration + 1,
        model=tbl,
        vary_size=vary_size
    )

    return ebb_mixture