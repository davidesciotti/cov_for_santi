import gc
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import json
import yaml
import scipy

import matplotlib

matplotlib.use("Qt5Agg")

sys.path.append('../lib')
import utils

sys.path.append('/Users/davide/Documents/Lavoro/Programmi/common_data/common_lib')
import my_module as mm

# ! settings
# import the yaml config file
# with open('../config/config.yaml', 'r') as ymlfile:
#     cfg = yaml.load(ymlfile)

# survey_area = cfg['survey_area']  # deg^2
survey_area = 14_700
deg2_in_sphere = 41252.96
fsky = survey_area / deg2_in_sphere

zbins = 13
ell_min = 10
ell_max = 5000

n_gal = np.genfromtxt('/Users/davide/Documents/Lavoro/Programmi/CLOE_validation/data/nuisance/nuiTabSPV3.dat')[:, 1]
sigma_eps = 0.26 * np.sqrt(2)

EP_or_ED = 'EP'
GL_or_LG = 'GL'
triu_tril = 'triu'
row_col_major = 'row-major'
probe_ordering = [['L', 'L'], [GL_or_LG[0], GL_or_LG[1]], ['G', 'G']]
block_index = 'ell'
n_probes = 2
input_folder = '../input/cl_LiFE/WeightedTest'
output_folder = '../output/WeightedTest'
plot_cls_tocheck = False
plot_differences = False
cosmology_id = 13
# ! end settings

cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, zbins)]

ind = mm.build_full_ind(triu_tril, row_col_major, zbins)
zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

ind_auto = ind[:zpairs_auto, :]
ind_cross = ind[zpairs_auto:zpairs_auto + zpairs_cross, :]

for cosmology_id in range(13, 23):
    for nbl in (32,):
        for weight_id in range(1, 3):

            # ! ell binning
            ell_values, delta_values, ell_bin_edges = utils.compute_ells(nbl, ell_min, ell_max, recipe='ISTF',
                                                                         output_ell_bin_edges=True)
            ell_bin_lower_edges = ell_bin_edges[:-1]
            ell_bin_upper_edges = ell_bin_edges[1:]

            # compare with the ones in the folder
            ell_values_input_file = np.genfromtxt(f'{input_folder}/NoEll{nbl:03d}/elltab.dat')
            ell_low = ell_values_input_file[:, 0]
            ell_up = ell_values_input_file[:, 1]
            ell_values_input = ell_values_input_file[:, 2]
            delta_ell_input = ell_values_input_file[:, 3]

            assert np.allclose(ell_values, ell_values_input), 'ell values dont match'
            assert np.allclose(delta_values, delta_ell_input), 'delta ell values dont match'

            # ! import, split and reshape the 1d vector
            cl_3x2pt_1d_w00 = np.genfromtxt(
                f'{input_folder}/NoEll{nbl:03d}/dv-3x2pt-LiFE-C{cosmology_id}-W00.dat')  # reference
            cl_3x2pt_1d = np.genfromtxt(
                f'{input_folder}/NoEll{nbl:03d}/dv-3x2pt-LiFE-C{cosmology_id}-W{weight_id:02d}.dat')

            elements_auto = nbl * zpairs_auto
            elements_cross = nbl * zpairs_cross
            elements_3x2pt = nbl * zpairs_3x2pt
            elements_2x2pt = elements_cross + elements_auto  # GGL + GCph

            assert len(cl_3x2pt_1d) == 2 * elements_auto + elements_cross == elements_3x2pt, \
                'wrong number of elements in the 3x2pt 1d data vector'

            cl_3x2pt_2d_w00 = cl_3x2pt_1d_w00.reshape((nbl, zpairs_3x2pt))
            cl_3x2pt_2d = cl_3x2pt_1d.reshape((nbl, zpairs_3x2pt))

            # split into 3 2d datavectors
            cl_ll_3x2pt_2d_w00 = cl_3x2pt_2d_w00[:, :zpairs_auto]
            cl_gl_3x2pt_2d_w00 = cl_3x2pt_2d_w00[:, zpairs_auto:zpairs_auto + zpairs_cross]
            cl_gg_3x2pt_2d_w00 = cl_3x2pt_2d_w00[:, zpairs_auto + zpairs_cross:]
            cl_ll_3x2pt_2d = cl_3x2pt_2d[:, :zpairs_auto]
            cl_gl_3x2pt_2d = cl_3x2pt_2d[:, zpairs_auto:zpairs_auto + zpairs_cross]
            cl_gg_3x2pt_2d = cl_3x2pt_2d[:, zpairs_auto + zpairs_cross:]

            cl_ll_3x2pt_3d_w00 = mm.cl_2D_to_3D_symmetric(cl_ll_3x2pt_2d_w00, nbl=nbl, zpairs=zpairs_auto, zbins=zbins)
            cl_gl_3x2pt_3d_w00 = mm.cl_2D_to_3D_asymmetric(cl_gl_3x2pt_2d_w00, nbl=nbl, zbins=zbins, order='C')
            cl_gg_3x2pt_3d_w00 = mm.cl_2D_to_3D_symmetric(cl_gg_3x2pt_2d_w00, nbl=nbl, zpairs=zpairs_auto, zbins=zbins)
            cl_ll_3x2pt_3d = mm.cl_2D_to_3D_symmetric(cl_ll_3x2pt_2d, nbl=nbl, zpairs=zpairs_auto, zbins=zbins)
            cl_gl_3x2pt_3d = mm.cl_2D_to_3D_asymmetric(cl_gl_3x2pt_2d, nbl=nbl, zbins=zbins, order='C')
            cl_gg_3x2pt_3d = mm.cl_2D_to_3D_symmetric(cl_gg_3x2pt_2d, nbl=nbl, zpairs=zpairs_auto, zbins=zbins)

            cl_3x2pt_5D_w00 = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
            cl_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))

            cl_3x2pt_5D_w00[0, 0, :, :, :] = cl_ll_3x2pt_3d_w00
            cl_3x2pt_5D_w00[1, 1, :, :, :] = cl_gg_3x2pt_3d_w00
            cl_3x2pt_5D_w00[1, 0, :, :, :] = cl_gl_3x2pt_3d_w00
            cl_3x2pt_5D_w00[0, 1, :, :, :] = np.transpose(cl_gl_3x2pt_3d_w00, (0, 2, 1))

            cl_3x2pt_5D[0, 0, :, :, :] = cl_ll_3x2pt_3d
            cl_3x2pt_5D[1, 1, :, :, :] = cl_gg_3x2pt_3d
            cl_3x2pt_5D[1, 0, :, :, :] = cl_gl_3x2pt_3d
            cl_3x2pt_5D[0, 1, :, :, :] = np.transpose(cl_gl_3x2pt_3d, (0, 2, 1))

            # ! quick check
            if plot_cls_tocheck:
                cl_LL_3D_old = np.load(f'../input/march_2023/Cl_LL.npy')
                cl_GL_3D_old = np.load(f'../input/march_2023/Cl_GL.npy')
                cl_GG_3D_old = np.load(f'../input/march_2023/Cl_GG.npy')
                ell_values_old = np.load(f'../input/ell_values.npy')
                delta_ell_old = np.load(f'../input/delta_ells.npy')

                zi, zj = 0, 0
                plt.figure()
                plt.loglog(ell_values_old, cl_LL_3D_old[:, zi, zj], label='old', marker='.')
                plt.loglog(ell_values, cl_3x2pt_5D[0, 0, :, zi, zj], label='new', marker='.')
                plt.legend()

            # ! Compute covariance
            if cosmology_id == 13:
                # create a noise with dummy axis for ell, to have the same shape as cl_3x2pt_5D
                noise_3x2pt_4D = mm.build_noise(zbins, n_probes, sigma_eps2=sigma_eps ** 2, ng=n_gal, EP_or_ED=EP_or_ED)
                noise_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
                for probe_A in (0, 1):
                    for probe_B in (0, 1):
                        for ell_idx in range(nbl):
                            noise_3x2pt_5D[probe_A, probe_B, ell_idx, :, :] = noise_3x2pt_4D[probe_A, probe_B, ...]

                # compute
                cov_3x2pt_10D_arr_w00 = mm.covariance_einsum(cl_3x2pt_5D_w00, noise_3x2pt_5D, fsky, ell_values,
                                                             delta_values)
                cov_3x2pt_10D_arr = mm.covariance_einsum(cl_3x2pt_5D, noise_3x2pt_5D, fsky, ell_values, delta_values)

                # reshape to 4D
                cov_3x2pt_10D_dict_w00 = mm.cov_10D_array_to_dict(cov_3x2pt_10D_arr_w00)
                cov_3x2pt_10D_dict = mm.cov_10D_array_to_dict(cov_3x2pt_10D_arr)
                cov_3x2pt_4D_w00 = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_10D_dict_w00, probe_ordering,
                                                               nbl, zbins, ind.copy(), GL_or_LG)
                cov_3x2pt_4D = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_10D_dict, probe_ordering,
                                                           nbl, zbins, ind.copy(), GL_or_LG)

                del cov_3x2pt_10D_dict_w00, cov_3x2pt_10D_dict, cov_3x2pt_10D_arr_w00, cov_3x2pt_10D_arr
                gc.collect()

                # reshape to 2D
                cov_3x2pt_2D_w00 = mm.cov_4D_to_2D(cov_3x2pt_4D_w00, block_index=block_index)
                cov_3x2pt_2D = mm.cov_4D_to_2D(cov_3x2pt_4D, block_index=block_index)
                # cov_3x2pt_w00_2DCLOE = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_w00_4D, nbl, zbins)
                # cov_3x2pt_2DCLOE = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_4D, nbl, zbins)

                np.save(f'../output/WeightedTest/NoEll{nbl:03d}/cov_3x2pt_2D_w00_{weight_id:02d}_C{cosmology_id}.npy',
                        cov_3x2pt_2D_w00)
                np.save(f'../output/WeightedTest/NoEll{nbl:03d}/cov_3x2pt_2D_{weight_id:02d}_C{cosmology_id}.npy',
                        cov_3x2pt_2D)

            else:
                cov_3x2pt_2D_w00 = np.load(
                    f'../output/WeightedTest/NoEll{nbl:03d}/cov_3x2pt_2D_w00_{weight_id:02d}_C13.npy')
                cov_3x2pt_2D = np.load(
                    f'../output/WeightedTest/NoEll{nbl:03d}/cov_3x2pt_2D_{weight_id:02d}_C13.npy')

            # ! Compute differences
            sigmas_w00 = np.sqrt(np.diag(cov_3x2pt_2D_w00))
            smape = np.zeros(len(cl_3x2pt_1d))
            diff_sigma = np.zeros(len(cl_3x2pt_1d))
            for k in range(len(cl_3x2pt_1d)):
                smape[k] = mm.compute_smape(cl_3x2pt_1d_w00[k], cl_3x2pt_1d[k])
                diff_sigma[k] = mm.compute_diff_sigma(cl_3x2pt_1d_w00[k], cl_3x2pt_1d[k], sigmas_w00[k])

            old_diff = mm.percent_diff(cl_3x2pt_1d, cl_3x2pt_1d_w00)

            if plot_differences:
                plt.figure()
                plt.title(f'case W{weight_id:02d}, C{cosmology_id}, nbl{nbl}')
                plt.plot(smape, label='SMAPE, uniform weights')
                plt.plot(diff_sigma, label='abs(diff)/sigma_W00')
                plt.plot(old_diff, label='old_diff')
                # plt.plot(cl_3x2pt_1d_w00, label='SMAPE, uniform weights')
                # plt.plot(cl_3x2pt_1d, label='abs(diff)/sigma_w00')
                plt.ylabel('difference [%]')
                plt.xlabel('datavector element')
                plt.legend()

            header = 'smape [%] \t abs(diff)/sigma_W00 [%]'
            results_tosave = np.column_stack((smape, diff_sigma))
            np.savetxt(
                f'../output/WeightedTest/NoEll{nbl:03d}/cl_difference-C{cosmology_id}-W00-vs-W{weight_id:02d}.dat',
                results_tosave, header=header)

            # ! Compute chi2


            cov_wl_1D_w00 = [0] * nbl
            for ell_bin in range(nbl):
                start = ell_bin * zpairs_3x2pt
                stop = start + zpairs_auto
                print(ell_bin, start, stop)
                cov_wl_1D_w00[ell_bin] = cov_3x2pt_2D_w00[start:stop, start:stop]

            cov_wl_2D_w00 = scipy.linalg.block_diag(*cov_wl_1D_w00)



            mm.matshow(cov_wl_2D_w00, log=True)
            assert False



            i = 0
            for ell_bin in range(nbl):
                for ij in range(zpairs_auto):
                    cl_wl_1d_w00[i] = cl_3x2pt_1d_w00[ij + ell_bin * zpairs_3x2pt]
                    i += 1

            inv_cov_3x2pt_2D_w00 = np.linalg.inv(cov_3x2pt_2D_w00)
            chi2_3x2pt_w00 = cl_3x2pt_1d_w00 @ inv_cov_3x2pt_2D_w00 @ cl_3x2pt_1d_w00

            if weight_id == 0 and nbl == 32:
                try:
                    cov_bench = np.genfromtxt(
                        f'../input/cl_LiFE/WeightedTest/NoEll{nbl:03d}/cm-3x2pt-LiFE-C{cosmology_id}-W{weight_id:02d}.dat')

                    assert np.allclose(cov_bench, cov_3x2pt_2D, rtol=1e-4,
                                       atol=0), 'covariance matrix is not the same as vincenzo\'s'
                except FileNotFoundError:
                    print(f'no benchmark covariance matrix found for case nbl{nbl}-C{cosmology_id}-W{weight_id:02d}')

            other_quantities_tosave = {
                'n_gal [arcmin^{-2}]': list(n_gal),
                'survey_area [deg^2]': survey_area,
                'sigma_eps': sigma_eps,
            }

            # np.savetxt(f'../output/cl_LiFE/WeightedTest/NoEll{nbl:03d}/CovMat-3x2pt-Gauss-{nbl}Bins.txt', cov_3x2pt_2DCLOE)
            # np.savetxt(f'{output_folder}/NoEll{nbl:03d}/cm-3x2pt-LiFE-C{cosmology_id}-W{weight_id:02d}.dat', cov_3x2pt_2D,
            #            fmt='%.8e')

            ell_grid_header = f'ell_bins = {nbl}\tell_min = {ell_min}\tell_max = {ell_max}\n' \
                              f'ell_bin_lower_edge\tell_bin_upper_edge\tell_bin_center\tdelta_ell'
            ell_grid = np.column_stack((ell_bin_lower_edges, ell_bin_upper_edges, ell_values, delta_values))
            np.savetxt(f'{output_folder}/NoEll{nbl:03d}/ell_grid.txt', ell_grid, header=ell_grid_header)

            with open(f'{output_folder}/NoEll{nbl:03d}/other_specs.txt', 'w') as file:
                file.write(json.dumps(other_quantities_tosave))

            del cov_3x2pt_2D
            gc.collect()

            print(f'case {nbl} ell bins, W{weight_id:02d}, cosmology C{cosmology_id} done')
