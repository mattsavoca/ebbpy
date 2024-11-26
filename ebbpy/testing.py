import polars as pl
import numpy as np
from scipy.stats import beta

def h(a, b, c, d, approx=False, log_h=False):
    """
    Calculate the probability that Beta(a, b) > Beta(c, d).
    """
    if approx:
        # Use normal approximation to the beta distribution
        u1 = a / (a + b)
        u2 = c / (c + d)
        var1 = a * b / ((a + b) ** 2 * (a + b + 1))
        var2 = c * d / ((c + d) ** 2 * (c + d + 1))
        std = np.sqrt(var1 + var2)
        z = (u1 - u2) / std
        ret = beta.cdf(z, 0, 1)
    else:
        # Exact computation using the definition
        # Note: For integer c, we can compute exactly; otherwise, we may need to use numerical integration
        # Here we use a numerical approximation
        from scipy.integrate import quad

        def integrand(x):
            return beta.pdf(x, a, b) * beta.cdf(x, c, d)

        ret, _ = quad(integrand, 0, 1)
    if log_h:
        return np.log(ret)
    else:
        return ret

def add_ebb_prop_test(tbl, threshold, alternative='greater', sort=False, approx=False):
    """
    Add columns representing a per-observation proportion test.

    Parameters:
    - tbl: Polars DataFrame that includes .alpha1 and .beta1 columns.
    - threshold: The proportion to which each observation is compared.
                 Can be a scalar or a tuple/list of two elements.
    - alternative: 'greater' or 'less'.
    - sort: Whether to sort the table in ascending order of posterior error probability.
    - approx: Whether to use a normal approximation to the beta distribution.
              Used only when comparing to another beta distribution.
    """
    if not isinstance(tbl, pl.DataFrame):
        raise TypeError("Input to add_ebb_prop_test must be a Polars DataFrame.")

    if not {'.alpha1', '.beta1'}.issubset(tbl.columns):
        raise ValueError("Input DataFrame must have '.alpha1' and '.beta1' columns.")

    if isinstance(threshold, (int, float)):
        # Compute PEP using beta CDF
        tbl = tbl.with_columns(
            pl.struct(['.alpha1', '.beta1']).apply(
                lambda row: beta.cdf(threshold, row['.alpha1'], row['.beta1'])
            ).alias('.pep')
        )
    elif isinstance(threshold, (tuple, list)) and len(threshold) == 2:
        # Compute PEP using the h function
        a, b = threshold
        tbl = tbl.with_columns(
            pl.struct(['.alpha1', '.beta1']).apply(
                lambda row: h(a, b, row['.alpha1'], row['.beta1'], approx=approx)
            ).alias('.pep')
        )
    else:
        raise ValueError("Threshold should be a scalar or a tuple/list of length 2.")

    if alternative == 'less':
        tbl = tbl.with_columns(
            (1 - pl.col('.pep')).alias('.pep')
        )
    elif alternative != 'greater':
        raise ValueError("Alternative must be 'greater' or 'less'.")

    # Compute q-values
    # Sort .pep in ascending order
    pep_df = tbl.select('.pep').sort('.pep')
    # Compute cumulative mean of sorted PEPs
    pep_values = pep_df['.pep'].to_numpy()
    cummean_pep = np.cumsum(pep_values) / np.arange(1, len(pep_values) + 1)
    # Map the cumulative means back to the original order
    pep_rank = tbl.select(pl.col('.pep').rank('dense')).to_series().to_numpy().astype(int) - 1
    qvalues = cummean_pep[pep_rank]
    tbl = tbl.with_columns(
        pl.Series('.qvalue', qvalues)
    )

    if sort:
        tbl = tbl.sort('.pep')

    return tbl
