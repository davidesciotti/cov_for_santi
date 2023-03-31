# 3x2pt covariance

In the repository you can find the code and covariances for the 3x2pt. Below are some clarifications on the conventions, which must be consistent for the data vector and covariance matrix (otherwise the $\chi^2$ will return meaningless values):

* `probe_ordering`: the ordering of the probes. I assumed the stacking (LL, GL, GG), which is the "default".
GL_or_LG: whether to use $C^{GL}{ij}(\ell)$ or $C^{LG}{ij}(\ell)$ (remember that $C^{LG}{ij}(\ell) = C^{GL}{ji}(\ell)$, i.e. you just need to transpose the redshift indices). Here, too, we have recently been using GL as the default, but if you want to change it, you can simply modify this string.
* `triu_tril`, `row_col_major`: for the auto-spectra (LL and GG), which are symmetric in $i, j$, choose whether to take only the upper or lower triangular part ("upper/lower triangle, `triu`/`tril`") row by row or column by column (i.e. with a ordering of row-major or col-major).
* `block_index`: in the 2D covariance matrix (which, in the end, is the only file that matters to you; I have included the others for completeness), the diagonal blocks correspond to a pair $(\ell_1, \ell_2)$. The fact that the matrix is block-diagonal comes from the fact that, for the Gaussian part, there is no covariance for $\ell_1 \neq \ell_2$. The alternative to the value ell is zpair, which means that the blocks are indexed by pairs of redshifts: $(zpair_i, zpair_j)$; the concept is the same but it is a bit less intuitive. However, to get an idea of what the 2D matrix looks like, you can use plt.matshow(cov_2D), maybe giving it the log of the covariance.

The 3x2pt covariance in 4D has shape (nbl, nbl, zpairs_3x2pt, zpairs_3x2pt), with `zpairs_3x2pt = zpairs_auto + zpairs_cross + zpairs_auto` for (LL, GL, GG) (= 55 + 100 + 55 = 210).

The 3x2pt covariance in 10D has shape (n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins), with n_probes = 2.

Note that I save the files in npz format. To load them, use:

    cov = np.load(filepath)['arr_0']
