import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Qt5Agg')

sys.path.append('../lib')
import my_module as mm


def build_X_matrix_BNT(BNT_matrix):
    """
    Builds the X matrix for the BNT transform, according to eq.
    :param BNT_matrix:
    :return:
    """
    X = {}
    delta_kron = np.eye(BNT_matrix.shape[0])
    X['L', 'L'] = np.einsum('ae, bf -> aebf', BNT_matrix, BNT_matrix)
    X['G', 'G'] = np.einsum('ae, bf -> aebf', delta_kron, delta_kron)
    X['G', 'L'] = np.einsum('ae, bf -> aebf', delta_kron, BNT_matrix)
    X['L', 'G'] = np.einsum('ae, bf -> aebf', BNT_matrix, delta_kron)
    return X


def cov_BNT_transform(cov_noBNT_6D, X_dict, probe_A, probe_B, probe_C, probe_D, optimize=True):
    """same as above, but only for one probe (i.e., LL or GL: GG is not modified by the BNT)"""
    # todo it's nicer if you sandwitch the covariance, maybe? That is, X cov X instead of X X cov
    cov_BNT_6D = np.einsum('aebf, cgdh, LMefgh -> LMabcd', X_dict[probe_A, probe_B], X_dict[probe_C, probe_D],
                           cov_noBNT_6D, optimize=optimize)
    return cov_BNT_6D


def cov_3x2pt_BNT_transform(cov_3x2pt_10D_dict, X_dict, optimize=True):
    """in np.einsum below, L and M are the ell1, ell2 indices, which are not touched by the BNT transform"""

    cov_3x2pt_BNT_dict_10D = {}

    for probe_A, probe_B, probe_C, probe_D in cov_3x2pt_10D_dict.keys():
        cov_3x2pt_BNT_dict_10D[probe_A, probe_B, probe_C, probe_D] = \
            cov_BNT_transform(cov_3x2pt_10D_dict[probe_A, probe_B, probe_C, probe_D], X_dict,
                              probe_A, probe_B, probe_C, probe_D, optimize=optimize)

    return cov_3x2pt_BNT_dict_10D


def cl_BNT_transform(cl_3D, BNT_matrix, probe_A, probe_B):
    assert cl_3D.ndim == 3, 'cl_3D must be 3D'
    assert BNT_matrix.ndim == 2, 'BNT_matrix must be 2D'
    assert cl_3D.shape[1] == BNT_matrix.shape[0], 'the number of ell bins in cl_3D and BNT_matrix must be the same'

    BNT_transform_dict = {
        'L': BNT_matrix,
        'G': np.eye(BNT_matrix.shape[0]),
    }

    cl_3D_BNT = np.zeros(cl_3D.shape)
    for ell_idx in range(cl_3D.shape[0]):
        cl_3D_BNT[ell_idx, :, :] = BNT_transform_dict[probe_A] @ \
                                   cl_3D[ell_idx, :, :] @ \
                                   BNT_transform_dict[probe_B].T

    return cl_3D_BNT


def cl_BNT_transform_3x2pt(cl_3x2pt_5D, BNT_matrix):
    """wrapper function to quickly implement the cl (or derivatives) BNT transform for the 3x2pt datavector"""

    cl_3x2pt_5D_BNT = np.zeros(cl_3x2pt_5D.shape)
    cl_3x2pt_5D_BNT[:, 0, 0, :, :] = cl_BNT_transform(cl_3x2pt_5D[:, 0, 0, :, :], BNT_matrix, 'L', 'L')
    cl_3x2pt_5D_BNT[:, 0, 1, :, :] = cl_BNT_transform(cl_3x2pt_5D[:, 0, 1, :, :], BNT_matrix, 'L', 'G')
    cl_3x2pt_5D_BNT[:, 1, 0, :, :] = cl_BNT_transform(cl_3x2pt_5D[:, 1, 0, :, :], BNT_matrix, 'G', 'L')
    cl_3x2pt_5D_BNT[:, 1, 1, :, :] = cl_3x2pt_5D[:, 1, 1, :, :]  # no need to transform the GG part

    return cl_3x2pt_5D_BNT


# ! settings
survey_area_ISTF = 15_000  # deg^2
deg2_in_sphere = 41252.96  # deg^2 in a spere

fsky = survey_area_ISTF / deg2_in_sphere
zbins = 10
nbl = 20
n_gal = 30
sigma_eps = 0.3
EP_or_ED = 'EP'
GL_or_LG = 'GL'
triu_tril = 'triu'
row_col_major = 'row-major'
probe_ordering = [['L', 'L'], [GL_or_LG[0], GL_or_LG[1]], ['G', 'G']]
block_index = 'ell'
n_probes = 2
input_folder = 'march_2023'
# ! end settings

ind = mm.build_full_ind(triu_tril, row_col_major, zbins)
zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

cl_LL_3D = np.load(f'../input/march_2023/Cl_LL.npy')
cl_GL_3D = np.load(f'../input/march_2023/Cl_GL.npy')
cl_GG_3D = np.load(f'../input/march_2023/Cl_GG.npy')

ell_values = np.load(f'../input/ell_values.npy')
delta_ell = np.load(f'../input/delta_ells.npy')
BNT_matrix = np.genfromtxt('../input/BNT_matrix.txt')

cl_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
cl_3x2pt_5D[0, 0, :, :, :] = cl_LL_3D
cl_3x2pt_5D[0, 1, :, :, :] = cl_GL_3D.transpose(0, 2, 1)
cl_3x2pt_5D[1, 0, :, :, :] = cl_GL_3D
cl_3x2pt_5D[1, 1, :, :, :] = cl_GG_3D

# create a noise with dummy axis for ell, to have the same shape as cl_3x2pt_5D
noise_3x2pt_4D = mm.build_noise(zbins, n_probes, sigma_eps2=sigma_eps ** 2, ng=n_gal, EP_or_ED=EP_or_ED)
noise_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
for probe_A in (0, 1):
    for probe_B in (0, 1):
        for ell_idx in range(nbl):
            noise_3x2pt_5D[probe_A, probe_B, ell_idx, :, :] = noise_3x2pt_4D[probe_A, probe_B, ...]

# 5d versions of auto.probe data and spectra
cl_LL_5D = cl_LL_3D[np.newaxis, np.newaxis, ...]
noise_LL_5D = noise_3x2pt_5D[0, 0, ...][np.newaxis, np.newaxis, ...]

# =================== no-BNT covariance ===============================
cov_WL_6D = mm.covariance_einsum(cl_LL_5D, noise_LL_5D, fsky, ell_values, delta_ell)[0, 0, 0, 0, ...]
cov_3x2pt_10D_arr = mm.covariance_einsum(cl_3x2pt_5D, noise_3x2pt_5D, fsky, ell_values, delta_ell)

# reshape to 4D
cov_WL_4D = mm.cov_6D_to_4D(cov_WL_6D, nbl, zpairs_auto, ind[:zpairs_auto])
cov_3x2pt_10D_dict = mm.cov_10D_array_to_dict(cov_3x2pt_10D_arr)  # not important, equivalent to the array above
cov_3x2pt_4D = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_10D_dict, probe_ordering, nbl, zbins, ind.copy(), GL_or_LG)

# reshape to 2D
cov_WL_2D = mm.cov_4D_to_2D(cov_WL_4D, block_index=block_index)
cov_3x2pt_2D = mm.cov_4D_to_2D(cov_3x2pt_4D, block_index=block_index)
cov_3x2pt_2DCLOE = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_4D, nbl, zbins)

# =================== BNT covariance ===================================
# transform (i need the 3x2pt dictionary for the BNT covariance)
X_dict = build_X_matrix_BNT(BNT_matrix)
cov_WL_BNTcov_6D = cov_BNT_transform(cov_WL_6D, X_dict, 'L', 'L', 'L', 'L')
cov_3x2pt_BNTcov_10D_dict = cov_3x2pt_BNT_transform(cov_3x2pt_10D_dict, X_dict)

# reshape to 4D
cov_WL_BNTcov_4D = mm.cov_6D_to_4D(cov_WL_BNTcov_6D, nbl, zpairs_auto, ind[:zpairs_auto])
cov_3x2pt_BNTcov_4D = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_BNTcov_10D_dict, probe_ordering, nbl, zbins, ind.copy(),
                                                  GL_or_LG)
# reshape to 2D
cov_WL_BNTcov_2D = mm.cov_4D_to_2D(cov_WL_BNTcov_4D, block_index=block_index)
cov_3x2pt_BNTcov_2D = mm.cov_4D_to_2D(cov_3x2pt_BNTcov_4D, block_index=block_index)
cov_3x2pt_BNTcov_2DCLOE = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_BNTcov_4D, nbl, zbins)

# =================== covariance with BNT cls ============================
# transform cls AND NOISE
cl_LL_BNT_3D = cl_BNT_transform(cl_LL_3D, BNT_matrix, 'L', 'L')
noise_LL_BNT_3D = cl_BNT_transform(noise_LL_5D[0, 0, ...], BNT_matrix, 'L', 'L')
cl_3x2pt_BNT_5D = cl_BNT_transform_3x2pt(cl_3x2pt_5D, BNT_matrix)
noise_3x2pt_BNT_5D = cl_BNT_transform_3x2pt(noise_3x2pt_5D, BNT_matrix)

# compute cov
cl_LL_BNT_5D = cl_LL_BNT_3D[np.newaxis, np.newaxis, ...]
noise_LL_BNT_5D = noise_LL_BNT_3D[np.newaxis, np.newaxis, ...]
cov_WL_BNTcl_6D = mm.covariance_einsum(cl_LL_BNT_5D, noise_LL_BNT_5D, fsky, ell_values, delta_ell)[0, 0, 0, 0, ...]
cov_3x2pt_BNTcl_10D_arr = mm.covariance_einsum(cl_3x2pt_BNT_5D, noise_3x2pt_BNT_5D, fsky, ell_values, delta_ell)

# reshape to 4D
cov_WL_BNTcl_4D = mm.cov_6D_to_4D(cov_WL_BNTcl_6D, nbl, zpairs_auto, ind[:zpairs_auto])
cov_3x2pt_BNTcl_10D_dict = mm.cov_10D_array_to_dict(cov_3x2pt_BNTcl_10D_arr)  # not important, equivalent to the array above
cov_3x2pt_BNTcl_4D = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_BNTcl_10D_dict, probe_ordering, nbl, zbins, ind.copy(), GL_or_LG)

# reshape to 2D
cov_WL_BNTcl_2D = mm.cov_4D_to_2D(cov_WL_BNTcl_4D, block_index=block_index)
cov_3x2pt_BNTcl_2D = mm.cov_4D_to_2D(cov_3x2pt_BNTcl_4D, block_index=block_index)
cov_3x2pt_BNTcl_2DCLOE = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_BNTcl_4D, nbl, zbins)

mm.compare_arrays(cov_3x2pt_BNTcov_2D, cov_3x2pt_BNTcl_2D, 'cov_3x2pt_BNTcov_2D', 'cov_3x2pt_BNTcl_2D',
                  plot_array=True, log_array=True,
                  plot_diff=True, log_diff=False, plot_diff_threshold=5)


np.savez_compressed('../output/cov_3x2pt_10D_arr.npz', cov_3x2pt_10D_arr)
np.savez_compressed('../output/cov_3x2pt_4D.npz', cov_3x2pt_4D)
np.savez_compressed('../output/cov_3x2pt_2D.npz', cov_3x2pt_2D)
np.savez_compressed('../output/cov_3x2pt_4D.npz', cov_3x2pt_4D)
np.savez_compressed('../output/cov_3x2pt_2D.npz', cov_3x2pt_2D)
