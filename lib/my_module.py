import bz2
import sys
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.interpolate import interp1d
import pickle
import itertools
import os


###############################################################################


def test_folder_content_old(output_path, benchmarks_path, extension):
    """tests that the outputs do not change between the old and the new version"""
    old_dict = dict(get_kv_pairs(benchmarks_path, extension))
    new_dict = dict(get_kv_pairs(output_path, extension))

    # check if the dictionaries are empty
    assert len(old_dict) > 0, 'No files in the benchmarks path ❌'
    assert len(new_dict) > 0, 'No files in the output path ❌'

    assert old_dict.keys() == new_dict.keys(), 'The number of files or their names has changed ❌'

    for key in old_dict.keys():
        try:
            if extension == 'npz':
                np.array_equal(old_dict[key]['arr_0'], new_dict[key]['arr_0'])
            else:
                np.array_equal(old_dict[key], new_dict[key])
        except AssertionError:
            f'The file {benchmarks_path}/{key}.{extension} is different ❌'

    print('tests passed successfully: the new files are equal to the benchmarks ✅')


def test_folder_content(output_path, benchmarks_path, extension, verbose=False, rtol=1e-10):
    """Test if the files in the output folder are equal to the benchmark files.

    Parameters:
    output_path (str): The path to the folder containing the output files.
    benchmarks_path (str): The path to the folder containing the benchmark files.
    extension (str): The extension of the files to be tested.

    Returns:
    None.
    """
    old_files = os.listdir(benchmarks_path)
    new_files = os.listdir(output_path)

    if 'benchmarks' in new_files:
        new_files.remove('benchmarks')

    # ignore hidden files
    old_files = [file for file in old_files if not file.startswith('.')]
    new_files = [file for file in new_files if not file.startswith('.')]

    assert old_files, f"No files found in the benchmarks path: {benchmarks_path} ❌"
    assert new_files, f"No files found in the output path: {output_path} ❌"
    assert set(old_files) == set(new_files), f"The number or name of files in the benchmark folder and output " \
                                             f"folder do not match ❌\n" \
                                             f"files which do not match: {set(old_files) ^ set(new_files)}"

    print(f'\n**** testing files in folder: *****\n{output_path}\n')

    for file_name in old_files:
        old_file_path = os.path.join(benchmarks_path, file_name)
        new_file_path = os.path.join(output_path, file_name)

        try:
            if extension == 'npz':
                np.testing.assert_allclose(np.load(old_file_path)['arr_0'], np.load(new_file_path)['arr_0'],
                                           verbose=verbose, rtol=rtol, atol=0)
            elif extension == 'npy':
                np.testing.assert_allclose(np.load(old_file_path), np.load(new_file_path), verbose=verbose, rtol=rtol,
                                           atol=0)
            elif extension == 'txt' or extension == 'dat':
                np.testing.assert_allclose(np.genfromtxt(old_file_path), np.genfromtxt(new_file_path),
                                           verbose=verbose, rtol=rtol, atol=0)
            else:
                raise ValueError(f"Unknown extension: {extension}")
        except AssertionError as exc:
            print(f'\nFile {file_name} ❌:', exc)
        else:
            print(f'The file {file_name} is equal to the benchmarks ✅')

    return None


def is_increasing(arr):
    return np.all(np.diff(arr) > 0)


def pycharm_exit():
    assert 1 > 2, 'aborting execution'


def save_pickle(filename, obj):
    with open(f'{filename}', 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(filename):
    with open(f'{filename}', 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def save_compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'wb') as handle:
        pickle.dump(data, handle)


def load_compressed_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data


@njit
def percent_diff(array_1, array_2, abs_value=False):
    diff = (array_1 / array_2 - 1) * 100
    if abs_value:
        return np.abs(diff)
    else:
        return diff


@njit
def percent_diff_mean(array_1, array_2):
    """
    result is in "percent" units
    """
    mean = (array_1 + array_2) / 2.0
    diff = (array_1 / mean - 1) * 100
    return diff


@njit
def percent_diff_nan(array_1, array_2, eraseNaN=True, log=False, abs_val=False):
    if eraseNaN:
        diff = np.where(array_1 == array_2, 0, percent_diff(array_1, array_2))
    else:
        diff = percent_diff(array_1, array_2)
    if log:
        diff = np.log10(diff)
    if abs_val:
        diff = np.abs(diff)
    return diff


def diff_threshold_check(diff, threshold):
    boolean = np.any(np.abs(diff) > threshold)
    print(f"has any element of the arrays a disagreement > {threshold}%? ", boolean)


def compare_arrays(A, B, name_A='A', name_B='B', plot_diff=False, plot_array=False, log_array=False, log_diff=False,
                   abs_val=False, plot_diff_threshold=None, white_where_zero=True):
    if plot_diff or plot_array:
        assert A.ndim == 2 and B.ndim == 2, 'plotting is only implemented for 2D arrays'

    # white = to_rgb('white')
    # cmap = ListedColormap([white] + plt.cm.viridis(np.arange(plt.cm.viridis.N)))
    # # set the color for 0 values as white and all other values to the standard colormap
    # cmap = plt.cm.viridis
    # cmap.set_bad(color=white)

    if plot_diff:

        diff_AB = percent_diff_nan(A, B, eraseNaN=True, log=log_diff, abs_val=abs_val)
        diff_BA = percent_diff_nan(B, A, eraseNaN=True, log=log_diff, abs_val=abs_val)

        if not np.allclose(diff_AB, diff_BA, rtol=1e-3, atol=0):
            print('diff_AB and diff_BA have a relative difference of more than 1%')

        if plot_diff_threshold is not None:
            # take the log of the threshold if using the log of the precent difference
            if log_diff:
                plot_diff_threshold = np.log10(plot_diff_threshold)

            print(f'plotting the *absolute value* of the difference only where it is below the given threshold '
                  f'({plot_diff_threshold}%)')
            diff_AB = np.ma.masked_where(np.abs(diff_AB) < plot_diff_threshold, np.abs(diff_AB))
            diff_BA = np.ma.masked_where(np.abs(diff_BA) < plot_diff_threshold, np.abs(diff_BA))

        fig, ax = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
        im = ax[0].matshow(diff_AB)
        ax[0].set_title(f'(A/B - 1) * 100')
        fig.colorbar(im, ax=ax[0])

        im = ax[1].matshow(diff_BA)
        ax[1].set_title(f'(B/A - 1) * 100')
        fig.colorbar(im, ax=ax[1])

        fig.suptitle(f'log={log_diff}, abs={abs_val}')
        plt.show()

    if plot_array:
        A_toplot, B_toplot = A, B

        if abs_val:
            A_toplot, B_toplot = np.abs(A), np.abs(B)
        if log_array:
            A_toplot, B_toplot = np.log10(A), np.log10(B)

        fig, ax = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
        im = ax[0].matshow(A_toplot)
        ax[0].set_title(f'{name_A}')
        fig.colorbar(im, ax=ax[0])

        im = ax[1].matshow(B_toplot)
        ax[1].set_title(f'{name_B}')
        fig.colorbar(im, ax=ax[1])
        fig.suptitle(f'log={log_diff}, abs={abs_val}')
        plt.show()

    if np.array_equal(A, B):
        print('A and B are equal ✅')
        return

    for rtol in [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]:  # these are NOT percent units, see print below
        if np.allclose(A, B, rtol=rtol, atol=0):
            print(f'A and B are close within relative tolerance of {rtol * 100}%) ✅')
            return

    diff_AB = percent_diff_nan(A, B, eraseNaN=True, abs_val=True)
    higher_rtol = plot_diff_threshold  # in "percent" units
    if higher_rtol is None:
        higher_rtol = 5.0
    result_emoji = '❌'
    no_outliers = np.where(diff_AB > higher_rtol)[0].shape[0]
    additional_info = f'\nMax discrepancy: {np.max(diff_AB):.2f}%;' \
                      f'\nNumber of elements with discrepancy > {higher_rtol}%: {no_outliers}' \
                      f'\nFraction of elements with discrepancy > {higher_rtol}%: {no_outliers / diff_AB.size:.5f}'
    print(f'Are A and B different by less than {higher_rtol}%? {result_emoji} {additional_info}')


def compare_folder_content(path_A: str, path_B: str, filetype: str):
    """
    Compare the content of 2 folders. The files in folder A should be a subset of the files in folder B.
    """
    dict_A = dict(get_kv_pairs(path_A, filetype))
    dict_B = dict(get_kv_pairs(path_B, filetype))

    for key in dict_A.keys():
        if np.array_equal(dict_A[key], dict_B[key]):
            result_emoji = '✅'
        else:
            result_emoji = '❌'
        print(f'is {key} equal in both folders? {result_emoji}')


def namestr(obj, namespace):
    """ does not work with slices!!! (why?)"""
    return [name for name in namespace if namespace[name] is obj][0]


def plot_FM(array, style=".-"):
    name = namestr(array, globals())
    plt.plot(range(7), array, style, label=name)


################################################ Fisher Matrix utilities ################################################
def find_null_rows_cols_2D(array):
    """
    :param array:

    :return null_rows_idxs: list
        array of null rows/columns indices
    """
    assert array.ndim == 2, 'ndim should be <= 2; higher-dimensional case not yet implemented'
    null_rows_idxs = np.where(np.all(array == 0, axis=0))[0]
    null_cols_idxs = np.where(np.all(array == 0, axis=1))[0]

    assert np.array_equal(null_rows_idxs, null_cols_idxs), \
        'null rows and columns indices should be the same for Fisher matrices'

    if null_rows_idxs.shape[0] == 0:
        print('The input array has no null rows/columns')
        return None
    else:
        print(f'The input array had null rows and columns at indices {null_rows_idxs}')
        return null_rows_idxs


def remove_rows_cols_array2D(array, rows_idxs_to_remove):
    """
    Removes the *same* rows and columns from an array
    :param array: numpy.ndarray. input 2D array
    :param rows_idxs_to_remove: list. rows (and columns) to delete
    :return: array without null rows and columns
    """
    if rows_idxs_to_remove is None:
        warnings.warn('null_rows_idxs is None, returning the input array')
        return array

    if len(rows_idxs_to_remove) == 0:
        warnings.warn('null_rows_idxs is empty, returning the input array')
        return array

    assert array.ndim == 2, 'ndim should be <= 2; higher-dimensional case not yet implemented'
    array = np.delete(array, rows_idxs_to_remove, axis=0)
    array = np.delete(array, rows_idxs_to_remove, axis=1)
    return array


def remove_null_rows_cols_2D_copilot(array_2d):
    """
    Remove null rows and columns from a 2D array - version by GitHub Copilot
    """

    assert array_2d.ndim == 2, 'ndim should be <= 2; higher-dimensional case not yet implemented'
    array_2d = array_2d[~np.all(array_2d == 0, axis=1)]
    array_2d = array_2d[:, ~np.all(array_2d == 0, axis=0)]
    return array_2d


def mask_FM_null_rowscols(FM, params, fid):
    """
    Mask the Fisher matrix, fiducial values and parameter list deleting the null rows and columns
    :param FM: Fisher Matrix, 2D numpy array
    :return: masked FM, fiducial values and parameter list
    """
    null_idx = find_null_rows_cols_2D(FM)

    if null_idx is None:
        return FM, params, fid

    FM = remove_rows_cols_array2D(FM, null_idx)
    params = np.delete(params, obj=null_idx, axis=0)
    fid = np.delete(fid, obj=null_idx, axis=0)
    assert len(fid) == len(params), 'the fiducial values and parameter lists should have the same length'
    return FM, list(params), list(fid)


def mask_FM(FM, param_names_dict, fiducials_dict, params_tofix_dict, remove_null_rows_cols=True):
    """
    Trim the Fisher matrix to remove null rows/columns and/or fix nuisance parameters
    :param FM:
    :param remaining_param_names_list:
    :param fid:
    :param n_cosmo_params:
    :param kwargs:
    :return:
    """

    # join param_names_dict.values() into single list
    all_param_names_list = list(itertools.chain(*list(param_names_dict.values())))
    all_fiducials_list = list(itertools.chain(*list(fiducials_dict.values())))

    # TODO - add option to fix specific parameter
    # TODO  - test this!!
    idx_todelete = []
    for key in params_tofix_dict.keys():
        if params_tofix_dict[key]:
            _param_names_list = param_names_dict[key]
            param_idxs = [all_param_names_list.index(param_name) for param_name in _param_names_list]
            idx_todelete.append(param_idxs)

    # make a continuous list
    # idx_todelete = np.flatten(idx_todelete.flatten())
    idx_todelete = list(itertools.chain(*idx_todelete))

    if idx_todelete:
        FM = np.delete(FM, idx_todelete, axis=0)
        FM = np.delete(FM, idx_todelete, axis=1)
        remaining_param_names_list = np.delete(all_param_names_list, idx_todelete)
        remaining_fiducials_list = np.delete(all_fiducials_list, idx_todelete)
    else:
        remaining_param_names_list = all_param_names_list
        remaining_fiducials_list = all_fiducials_list

    # remove remaining null rows_cols
    if remove_null_rows_cols:
        FM, remaining_param_names_list, remaining_fiducials_list = mask_FM_null_rowscols(FM, remaining_param_names_list,
                                                                                         remaining_fiducials_list)

    return FM, list(remaining_param_names_list), list(remaining_fiducials_list)


def build_labels_TeX(zbins):
    galaxy_bias_label = ['$b_{%i}$' % (i + 1) for i in range(zbins)]
    shear_bias_label = ['$m_{%i}$' % (i + 1) for i in range(zbins)]
    zmean_shift_label = ['$dz_{%i}$' % (i + 1) for i in range(zbins)]
    return [galaxy_bias_label, shear_bias_label, zmean_shift_label]


def build_labels(zbins):
    galaxy_bias_label = ['b%i' % (i + 1) for i in range(zbins)]
    shear_bias_label = ['m%i' % (i + 1) for i in range(zbins)]
    zmean_shift_label = ['dz%i' % (i + 1) for i in range(zbins)]
    return [galaxy_bias_label, shear_bias_label, zmean_shift_label]


def matshow(array, title="title", log=False, abs_val=False, threshold=None):
    """
    :param array:
    :param title:
    :param log:
    :param abs_val:
    :param threshold: if None, do not mask the values; otherwise, keep only the elements above the threshold
    (i.e., mask the ones below the threshold)
    :return:
    """
    # the ordering of these is important: I want the log(abs), not abs(log)
    if abs_val:  # take the absolute value
        array = np.abs(array)
        title = 'abs ' + title
    if log:  # take the log
        array = np.log10(array)
        title = 'log10 ' + title

    if threshold is not None:
        array = np.ma.masked_where(array < threshold, array)
        title += f" (\nmasked below {threshold})"

    plt.matshow(array)
    plt.colorbar()
    plt.title(title)


# load txt or dat files in dictionary
def get_kv_pairs(path_import, extension='npy'):
    """
    to use it, wrap it in "dict(), e.g.:
        loaded_dict = dict(get_kv_pairs(path_import, filetype="dat"))
    """
    if extension == 'npy' or extension == 'npz':
        load_function = np.load
    elif extension == 'txt' or extension == 'dat':
        load_function = np.genfromtxt
    else:
        raise NotImplementedError("extension must be either 'npy', 'npz', 'txt' or 'dat'")

    for path in Path(path_import).glob(f"*.{extension}"):
        yield path.stem, load_function(str(path))


# to display the names (keys) more tidily
def show_keys(arrays_dict):
    for key in arrays_dict:
        print(key)


def cl_interpolator(cl_2D, zpairs, new_ell_values, nbl, kind='linear'):
    original_ell_values = cl_2D[:, 0]
    cl_interpolated = np.zeros((nbl, zpairs))
    for zpair_idx in range(zpairs):
        f = interp1d(original_ell_values, cl_2D[:, zpair_idx + 1], kind=kind)
        cl_interpolated[:, zpair_idx] = f(new_ell_values)
    return cl_interpolated


# def cl_interpolator_no_1st_column(npairs, cl_2D, original_ell_values, new_ell_values, nbl):
#     Cl_interpolated = np.zeros((nbl, npairs))
#     for j in range(npairs):
#         f = interp1d(original_ell_values, cl_2D[:, j], kind='linear')
#         Cl_interpolated[:, j] = f(new_ell_values)
#     return Cl_interpolated


@njit
def symmetrize_Cl(Cl, nbl, zbins):
    for ell in range(nbl):
        for i in range(zbins):
            for j in range(zbins):
                Cl[ell, j, i] = Cl[ell, i, j]
    return Cl


def generate_ind(triu_tril_square, row_col_major, size):
    """
    Generates a list of indices for the upper triangular part of a matrix
    :param triu_tril_square: str. if 'triu', returns the indices for the upper triangular part of the matrix.
    If 'tril', returns the indices for the lower triangular part of the matrix
    If 'full_square', returns the indices for the whole matrix
    :param row_col_major: str. if True, the indices are returned in row-major order; otherwise, in column-major order
    :param size: int. size of the matrix to take the indices of
    :return: list of indices
    """
    assert row_col_major in ['row-major', 'col-major'], 'row_col_major must be either "row-major" or "col-major"'
    assert triu_tril_square in ['triu', 'tril', 'full_square'], 'triu_tril_square must be either "triu", "tril" or ' \
                                                                '"full_square"'

    if triu_tril_square == 'triu':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(i, size)]
        elif 'col-major':
            ind = [(j, i) for i in range(size) for j in range(i + 1)]
    elif triu_tril_square == 'tril':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(i + 1)]
        elif 'col-major':
            ind = [(j, i) for i in range(size) for j in range(i, size)]
    elif triu_tril_square == 'full_square':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(size)]
        elif 'col-major':
            ind = [(j, i) for i in range(size) for j in range(size)]

    return np.asarray(ind)


def build_full_ind(triu_tril, row_col_major, size):
    """
    Builds the good old ind file
    """

    assert triu_tril in ['triu', 'tril'], 'triu_tril must be either "triu" or "tril"'
    assert row_col_major in ['row-major', 'col-major'], 'row_col_major must be either "row-major" or "col-major"'

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(size)

    LL_columns = np.zeros((zpairs_auto, 2))
    GL_columns = np.hstack((np.ones((zpairs_cross, 1)), np.zeros((zpairs_cross, 1))))
    GG_columns = np.ones((zpairs_auto, 2))

    LL_columns = np.hstack((LL_columns, generate_ind(triu_tril, row_col_major, size))).astype(int)
    GL_columns = np.hstack((GL_columns, generate_ind('full_square', row_col_major, size))).astype(int)
    GG_columns = np.hstack((GG_columns, generate_ind(triu_tril, row_col_major, size))).astype(int)

    ind = np.vstack((LL_columns, GL_columns, GG_columns))

    assert ind.shape[0] == zpairs_3x2pt, 'ind has the wrong number of rows'

    return ind


######## CHECK FOR DUPLICATES
def cl_2D_to_3D_symmetric(Cl_2D, nbl, zpairs, zbins):
    """ reshape from (nbl, zpairs) to (nbl, zbins, zbins) according to
    upper traigular ordering 0-th rows filled first, then second from i to 10...
    """
    triu_idx = np.triu_indices(zbins)
    Cl_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        for zpair_idx in range(zpairs):
            i, j = triu_idx[0][zpair_idx], triu_idx[1][zpair_idx]
            Cl_3D[ell, i, j] = Cl_2D[ell, zpair_idx]
    # fill lower diagonal (the matrix is symmetric!)
    Cl_3D = fill_3D_symmetric_array(Cl_3D, nbl, zbins)
    return Cl_3D


def cl_2D_to_3D_symmetric_bu(Cl_2D, nbl, zpairs, zbins):
    """ reshape from (nbl, zpairs) to (nbl, zbins, zbins) according to
    upper traigular ordering 0-th rows filled first, then second from i to 10...
    """
    triu_idx = np.triu_indices(zbins)
    Cl_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        for zpair_idx in range(zpairs):
            i, j = triu_idx[0][zpair_idx], triu_idx[1][zpair_idx]
            Cl_3D[ell, i, j] = Cl_2D[ell, zpair_idx]
    # fill lower diagonal (the matrix is symmetric!)
    Cl_3D = fill_3D_symmetric_array(Cl_3D, nbl, zbins)
    return Cl_3D


def cl_2D_to_3D_asymmetric(Cl_2D, nbl, zbins, order):
    """ reshape from (nbl, npairs) to (nbl, zbins, zbins), rows first
    (valid for asymmetric Cij, i.e. C_XC)
    """
    assert order in ['row-major', 'col-major', 'C', 'F'], 'order must be either "row-major", "C" (equivalently), or' \
                                                          '"col-major", "F" (equivalently)'
    if order == 'row-major':
        order = 'C'
    elif order == 'col-major':
        order = 'F'

    Cl_3D = np.zeros((nbl, zbins, zbins))
    Cl_3D = np.reshape(Cl_2D, Cl_3D.shape, order=order)
    return Cl_3D


def Cl_3D_to_2D_symmetric(Cl_3D, nbl, npairs, zbins=10):
    """ reshape from (nbl, zbins, zbins) to (nbl, npairs)  according to
    upper traigular ordering 0-th rows filled first, then second from i to 10...
    """
    triu_idx = np.triu_indices(zbins)
    Cl_2D = np.zeros((nbl, npairs))
    for ell in range(nbl):
        for i in range(npairs):
            Cl_2D[ell, i] = Cl_3D[ell, triu_idx[0][i], triu_idx[1][i]]
    return Cl_2D


def Cl_3D_to_2D_asymmetric(Cl_3D):
    """ reshape from (nbl, zbins, zbins) to (nbl, npairs), rows first 
    (valid for asymmetric Cij, i.e. C_XC)
    """
    assert Cl_3D.ndim == 3, 'Cl_3D must be a 3D array'

    nbl = Cl_3D.shape[0]
    zbins = Cl_3D.shape[1]
    zpairs_cross = zbins ** 2

    Cl_2D = np.reshape(Cl_3D, (nbl, zpairs_cross))

    # Cl_2D = np.zeros((nbl, zpairs_cross))
    # for ell in range(nbl):
    #     Cl_2D[ell, :] = Cl_3D[ell, :].flatten(order='C')
    return Cl_2D


@njit
def cl_1D_to_3D(cl_1d, nbl: int, zbins: int, is_symmetric: bool):
    """ This is used to unpack Vincenzo's files for SPV3
    Still to be thoroughly checked."""

    cl_3d = np.zeros((nbl, zbins, zbins))
    p = 0
    if is_symmetric:
        for ell in range(nbl):
            for iz in range(zbins):
                for jz in range(iz, zbins):
                    cl_3d[ell, iz, jz] = cl_1d[p]
                    p += 1

    else:  # take all elements, not just the upper triangle
        for ell in range(nbl):
            for iz in range(zbins):
                for jz in range(zbins):
                    cl_3d[ell, iz, jz] = cl_1d[p]
                    p += 1
    return cl_3d


# XXX NEW AND CORRECTED FUNCTIONS TO MAKE THE Cl 3D
###############################################################################
def array_2D_to_3D_ind(array_2D, nbl, zbins, ind, start, stop):
    # ! is this to be deprecated??
    """ unpack according to "ind" ordering the same as the Cl!! """
    print('attention, assuming npairs = 55 (that is, zbins = 10)!')
    array_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        for k, p in enumerate(range(start, stop)):
            array_3D[ell, ind[p, 2], ind[p, 3]] = array_2D[ell, k]
            # enumerate is in case p deosn't start from p, that is, for LG
    return array_3D


# @njit
def symmetrize_2d_array(array_2d):
    """ mirror the lower/upper triangle """

    # if already symmetric, do nothing
    if check_symmetric(array_2d, exact=True):
        return array_2d

    # there is an implicit "else" here, since the function returns array_2d if the array is symmetric
    assert array_2d.ndim == 2, 'array must be square'
    size = array_2d.shape[0]

    # check that either the upper or lower triangle (not including the diagonal) is null
    triu_elements = array_2d[np.triu_indices(size, k=+1)]
    tril_elements = array_2d[np.tril_indices(size, k=-1)]
    assert np.all(triu_elements) == 0 or np.all(tril_elements) == 0, 'neither the upper nor the lower triangle ' \
                                                                     '(excluding the diagonal) are null'

    assert np.any(np.diag(array_2d)) != 0, 'the diagonal elements are all null. ' \
                                           'This is not necessarily an error, but is suspect'

    # symmetrize
    array_2d = np.where(array_2d, array_2d, array_2d.T)
    # check
    if not check_symmetric(array_2d, exact=False):
        warnings.warn('check failed: the array is not symmetric')

    return array_2d


def fill_3D_symmetric_array(array_3D, nbl, zbins):
    """ mirror the lower/upper triangle """
    assert array_3D.shape == (nbl, zbins, zbins), 'shape of input array must be (nbl, zbins, zbins)'

    array_diag_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        array_diag_3D[ell, :, :] = np.diag(np.diagonal(array_3D, 0, 1, 2)[ell, :])
    array_3D = array_3D + np.transpose(array_3D, (0, 2, 1)) - array_diag_3D
    return array_3D


def array_2D_to_1D_ind(array_2D, zpairs, ind):
    """ unpack according to "ind" ordering, same as for the Cls """
    assert ind.shape[0] == zpairs, 'ind must have lenght zpairs'

    array_1D = np.zeros(zpairs)
    for p in range(zpairs):
        i, j = ind[p, 2], ind[p, 3]
        array_1D[p] = array_2D[i, j]
    return array_1D


###############################################################################


############### FISHER MATRIX ################################


# interpolator for FM
# XXXX todo
def interpolator(dC_interpolated_dict, dC_dict, obs_name, params_names, nbl, zpairs, ell_values, suffix):
    print('deprecated?')
    for param_name in params_names:  # loop for each parameter
        # pick array to interpolate
        dC_to_interpolate = dC_dict[f"dCij{obs_name}d{param_name}-{suffix}"]
        dC_interpolated = np.zeros((nbl, zpairs))  # initialize interpolated array

        # now interpolate
        original_ell_values = dC_to_interpolate[:, 0]  # first column is ell
        dC_to_interpolate = dC_to_interpolate[:, 1:]  # remove ell column
        for zpair_idx in range(zpairs):
            f = interp1d(original_ell_values, dC_to_interpolate[:, zpair_idx], kind='linear')
            dC_interpolated[:, zpair_idx] = f(ell_values)  # fill zpair_idx-th column
            dC_interpolated_dict[f"dCij{obs_name}d{param_name}-{suffix}"] = dC_interpolated  # store array in the dict

    return dC_interpolated_dict


# @njit
def fill_dC_array(params_names, dC_interpolated_dict, probe_code, dC, suffix):
    for (counter, param) in enumerate(params_names):
        dC[:, :, counter] = dC_interpolated_dict[f"dCij{probe_code}d{param}-{suffix}"]
    return dC


def fill_datavector_4D(nParams, nbl, npairs, zbins, ind, dC_4D):
    # XXX pairs_tot
    D_4D = np.zeros((nbl, zbins, zbins, nParams))

    for alf in range(nParams):
        for elle in range(nbl):
            for p in range(npairs):
                if ind[p, 0] == 0 and ind[p, 1] == 0:
                    D_4D[elle, ind[p, 2], ind[p, 3], alf] = dC_4D[elle, ind[p, 2], ind[p, 3], alf]
    return D_4D


@njit
def datavector_3D_to_2D(D_3D, nParams, nbl, npairs):
    D_2D = np.zeros((npairs * nbl, nParams))
    for alf in range(nParams):
        count = 0
        for elle in range(nbl):
            for p in range(npairs):
                D_2D[count, alf] = D_3D[elle, p, alf]
                count = count + 1
    return D_2D


# @njit
def compute_FM_3D(nbl, npairs, nParams, cov_inv, D_3D):
    """ Compute FM using 3D datavector - 2D + the cosmological parameter axis - and 3D covariance matrix (working but
    deprecated in favor of compute_FM_2D)"""
    b = np.zeros((nbl, npairs, nParams))
    FM = np.zeros((nParams, nParams))
    for alf in range(nParams):
        for bet in range(nParams):
            for elle in range(nbl):
                b[elle, :, bet] = cov_inv[elle, :, :] @ D_3D[elle, :, bet]
                FM[alf, bet] = FM[alf, bet] + (D_3D[elle, :, alf] @ b[elle, :, bet])
    return FM


# @njit
def compute_FM_2D(nbl, npairs, nparams_tot, cov_2D_inv, D_2D):
    """ Compute FM using 2D datavector - 1D + the cosmological parameter axis - and 2D covariance matrix"""
    b = np.zeros((nbl * npairs, nparams_tot))
    FM = np.zeros((nparams_tot, nparams_tot))
    for alf in range(nparams_tot):
        for bet in range(nparams_tot):
            b[:, bet] = cov_2D_inv[:, :] @ D_2D[:, bet]
            FM[alf, bet] = D_2D[:, alf] @ b[:, bet]
    return FM


def compute_FM_2D_optimized(nbl, npairs, nparams_tot, cov_2D_inv, D_2D):
    """ Compute FM using 2D datavector - 1D + the cosmological parameter axis - and 2D covariance matrix"""
    warnings.warn('deprecate this?')
    b = np.zeros((nbl * npairs, nparams_tot))
    FM = np.zeros((nparams_tot, nparams_tot))
    for alf in range(nparams_tot):
        for bet in range(nparams_tot):
            b[:, bet] = cov_2D_inv[:, :] @ D_2D[:, bet]
            FM[alf, bet] = D_2D[:, alf] @ b[:, bet]

    # do it with np.einsum in one line
    # FM = np.einsum('ij,ik,jk->ij', D_2D, b, cov_2D_inv)
    b = np.einsum('ij,jk->ik', cov_2D_inv, D_2D)
    FM = np.einsum('ij,jk->ik', D_2D, b)
    return FM


def compute_FoM(FM, w0wa_idxs=(2, 3)):
    print('rows/cols 2 and 3 for w0, wa')
    start = w0wa_idxs[0]
    stop = w0wa_idxs[1] + 1
    cov_param = np.linalg.inv(FM)
    cov_param_reduced = cov_param[start:stop, start:stop]
    FM_reduced = np.linalg.inv(cov_param_reduced)
    FoM = np.sqrt(np.linalg.det(FM_reduced))
    return FoM


def get_ind_file(path, ind_ordering, which_forecast):
    if ind_ordering == 'vincenzo' or which_forecast == 'sylvain':
        ind = np.genfromtxt(path.parent / "common_data/indici.dat").astype(int)
        ind = ind - 1
    elif ind_ordering == 'CLOE':
        ind = np.genfromtxt(path.parent / "common_data/indici_luca/indici_cloe_like.dat").astype(int)
        ind = ind - 1
    elif ind_ordering == 'SEYFERT':
        ind = np.genfromtxt(path.parent / "common_data/indici_luca/indici_seyfert_like.dat").astype(int)
        ind = ind - 1
    else:
        raise ValueError('ind_ordering must be vincenzo, sylvain, CLOE or SEYFERT')
    return ind


def get_output_folder(ind_ordering, which_forecast):
    if which_forecast == 'IST':
        if ind_ordering == 'vincenzo':
            output_folder = 'ISTspecs_indVincenzo'
        elif ind_ordering == 'CLOE':
            output_folder = 'ISTspecs_indCLOE'
        elif ind_ordering == 'SEYFERT':
            output_folder = 'ISTspecs_indSEYFERT'
    elif which_forecast == 'sylvain':
        output_folder = 'common_ell_and_deltas'
    return output_folder


def get_zpairs(zbins):
    zpairs_auto = int((zbins * (zbins + 1)) / 2)  # = 55 for zbins = 10, cast it as int
    zpairs_cross = zbins ** 2
    zpairs_3x2pt = 2 * zpairs_auto + zpairs_cross
    return zpairs_auto, zpairs_cross, zpairs_3x2pt


###############################################################################
#################### COVARIANCE MATRIX COMPUTATION ############################ 
###############################################################################
# TODO unify these 3 into a single function
# TODO workaround for start_index, stop_index (super easy)

@njit
def covariance(nbl, npairs, start_index, stop_index, Cij, noise, l_lin, delta_l, fsky, ind):
    # create covariance array
    covariance = np.zeros((nbl, nbl, npairs, npairs))
    # compute cov(ell, p, q)
    for ell in range(nbl):
        for p in range(start_index, stop_index):
            for q in range(start_index, stop_index):
                covariance[ell, ell, p - start_index, q - start_index] = \
                    ((Cij[ell, ind[p, 2], ind[q, 2]] + noise[ind[p, 0], ind[q, 0], ind[p, 2], ind[q, 2]]) *
                     (Cij[ell, ind[p, 3], ind[q, 3]] + noise[ind[p, 1], ind[q, 1], ind[p, 3], ind[q, 3]]) +
                     (Cij[ell, ind[p, 2], ind[q, 3]] + noise[ind[p, 0], ind[q, 1], ind[p, 2], ind[q, 3]]) *
                     (Cij[ell, ind[p, 3], ind[q, 2]] + noise[ind[p, 1], ind[q, 0], ind[p, 3], ind[q, 2]])) / \
                    ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])
    return covariance


def covariance_einsum(cl_5d, noise_5d, f_sky, ell_values, delta_ell, return_only_diagonal_ells=False):
    """
    computes the 10-dimensional covariance matrix, of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins). The 5-th axis is added only if
    return_only_diagonal_ells is True. *for the single-probe case, n_probes = 1*

    In np.einsum, the indices have the following meaning:
        A, B, C, D = probe identifier. 0 for WL, 1 for GCph
        L, M = ell, ell_prime
        i, j, k, l = redshift bin indices

    cl_5d must have shape = (n_probes, n_probes, nbl, zbins, zbins) = (A, B, L, i, j), same as noise_5d

    :param cl_5d:
    :param noise_5d:
    :param f_sky:
    :param ell_values:
    :param delta_ell:
    :param return_only_diagonal_ells:
    :return: 10-dimensional numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins), containing the covariance.

    """
    assert cl_5d.shape[0] == 1 or cl_5d.shape[0] == 2, 'This funcion only works with 1 or two probes'
    assert cl_5d.shape[0] == cl_5d.shape[1], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert cl_5d.shape[-1] == cl_5d.shape[-2], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert noise_5d.shape == cl_5d.shape, 'noise_5d must have shape the same shape as cl_5d, although there ' \
                                          'is no ell dependence'

    nbl = cl_5d.shape[2]

    prefactor = 1 / ((2 * ell_values + 1) * f_sky * delta_ell)

    # considering ells off-diagonal (wrong for Gauss: I am not implementing the delta)
    # term_1 = np.einsum('ACLik, BDMjl -> ABCDLMijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    # term_2 = np.einsum('ADLil, BCMjk -> ABCDLMijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    # cov_10d = np.einsum('ABCDLMijkl, L -> ABCDLMijkl', term_1 + term_2, prefactor)

    # considering only ell diagonal
    term_1 = np.einsum('ACLik, BDLjl -> ABCDLijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    term_2 = np.einsum('ADLil, BCLjk -> ABCDLijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    cov_9d = np.einsum('ABCDLijkl, L -> ABCDLijkl', term_1 + term_2, prefactor)

    if return_only_diagonal_ells:
        warnings.warn('return_only_diagonal_ells is True, the array will be 9-dimensional, potentially causing '
                      'problems when reshaping or summing to cov_SSC arrays')
        return cov_9d

    n_probes = cov_9d.shape[0]
    zbins = cov_9d.shape[-1]
    cov_10d = np.zeros((n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins))
    cov_10d[:, :, :, :, np.arange(nbl), np.arange(nbl), ...] = cov_9d[:, :, :, :, np.arange(nbl), ...]

    return cov_10d


def cov_10D_dict_to_array(cov_10D_dict, nbl, zbins, n_probes=2):
    """ transforms a dictionary of "shape" [(A, B, C, D)][nbl, nbl, zbins, zbins, zbins, zbins] (where A, B, C, D is a
    tuple of strings, each one being either 'L' or 'G') to a numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins)"""
    cov_10D_array = \
        np.zeros((n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins))

    LG_idx_dict = {'L': 0, 'G': 1}
    for A, B, C, D in cov_10D_dict.keys():
        cov_10D_array[LG_idx_dict[A], LG_idx_dict[B], LG_idx_dict[C], LG_idx_dict[D], ...] = \
            cov_10D_dict[A, B, C, D]

    return cov_10D_array


def cov_10D_array_to_dict(cov_10D_array, n_probes=2):
    """ transforms a dictionary of "shape" [(A, B, C, D)][nbl, nbl, zbins, zbins, zbins, zbins] (where A, B, C, D is a
    tuple of strings, each one being either 'L' or 'G') to a numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins)"""

    # cov_10D_dict = {}
    # for A in ('L', 'G'):
    #     for B in ('L', 'G'):
    #         for C in ('L', 'G'):
    #             for D in ('L', 'G'):
    #                 cov_10D_dict[A, B, C, D] = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))

    cov_10D_dict = {}
    LG_idx_tuple = ('L', 'G')
    for A in range(n_probes):
        for B in range(n_probes):
            for C in range(n_probes):
                for D in range(n_probes):
                    cov_10D_dict[LG_idx_tuple[A], LG_idx_tuple[B], LG_idx_tuple[C], LG_idx_tuple[D]] = \
                        cov_10D_array[A, B, C, D, ...]

    return cov_10D_dict


# @njit
def covariance_WA(nbl, npairs, start_index, stop_index, Cij, noise, l_lin, delta_l, fsky, ind, ell_WA):
    covariance = np.zeros((nbl, nbl, npairs, npairs))

    for ell in range(nbl):
        for p in range(start_index, stop_index):
            for q in range(start_index, stop_index):

                if ell_WA.size == 1:  # in the case of just one bin it would give error
                    denominator = ((2 * l_lin + 1) * fsky * delta_l)
                else:
                    denominator = ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])

                covariance[ell, ell, p - start_index, q - start_index] = \
                    ((Cij[ell, ind[p, 2], ind[q, 2]] + noise[ind[p, 0], ind[q, 0], ind[p, 2], ind[q, 2]]) * (
                            Cij[ell, ind[p, 3], ind[q, 3]] + noise[ind[p, 1], ind[q, 1], ind[p, 3], ind[q, 3]]) + (
                             Cij[ell, ind[p, 2], ind[q, 3]] + noise[ind[p, 0], ind[q, 1], ind[p, 2], ind[q, 3]]) * (
                             Cij[ell, ind[p, 3], ind[q, 2]] + noise[ind[p, 1], ind[q, 0], ind[p, 3], ind[q, 2]])) \
                    / denominator
    return covariance


# covariance matrix for ALL
@njit
def covariance_ALL(nbl, npairs, Cij, noise, l_lin, delta_l, fsky, ind):
    # create covariance array
    cov_GO = np.zeros((nbl, nbl, npairs, npairs))
    # compute cov(ell, p, q)
    for ell in range(nbl):
        for p in range(npairs):
            for q in range(npairs):
                # ind carries info about both the probes and the z indices!
                A, B, C, D = ind[p, 0], ind[p, 1], ind[q, 0], ind[q, 1]
                i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]

                cov_GO[ell, ell, p, q] = \
                    ((Cij[ell, A, C, i, k] + noise[A, C, i, k]) * (Cij[ell, B, D, j, l] + noise[B, D, j, l]) +
                     (Cij[ell, A, D, i, l] + noise[A, D, i, l]) * (Cij[ell, B, C, j, k] + noise[B, C, j, k])) / \
                    ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])
    return cov_GO


@njit
def cov_SSC_old(nbl, npairs, ind, Cij, Sijkl, fsky, probe, zbins, Rl):
    if probe in ["WL", "WA"]:
        shift = 0
    elif probe == "GC":
        shift = zbins
    else:
        raise ValueError('probe must be "WL", "WA" or "GC"')

    cov_SSC = np.zeros((nbl, nbl, npairs, npairs))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs):
                for q in range(npairs):
                    i, j = ind[p, 2], ind[p, 3]
                    k, l = ind[q, 2], ind[q, 3]

                    cov_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                 Cij[ell1, i, j] * Cij[ell2, k, l] *
                                                 Sijkl[i + shift, j + shift, k + shift, l + shift])
    cov_SSC /= fsky
    return cov_SSC


@njit
def cov_SSC(nbl, zpairs, ind, Cij, Sijkl, fsky, probe, zbins, Rl):
    if probe in ["WL", "WA"]:
        shift = 0
    elif probe == "GC":
        shift = zbins
    else:
        raise ValueError('probe must be "WL", "WA" or "GC"')

    cov_SSC = np.zeros((nbl, nbl, zpairs, zpairs))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(zpairs):
                for q in range(zpairs):
                    i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]

                    cov_SSC[ell1, ell2, p, q] = (Rl[ell1, i, j] * Rl[ell2, k, l] *
                                                 Cij[ell1, i, j] * Cij[ell2, k, l] *
                                                 Sijkl[i + shift, j + shift, k + shift, l + shift])
    cov_SSC /= fsky
    return cov_SSC


@njit
def build_Sijkl_dict(Sijkl, zbins):
    # build probe lookup dictionary, to set the right start and stop values
    probe_lookup = {
        'L': {
            'start': 0,
            'stop': zbins
        },
        'G': {
            'start': zbins,
            'stop': 2 * zbins
        }
    }

    # fill Sijkl dictionary
    Sijkl_dict = {}
    for probe_A in ['L', 'G']:
        for probe_B in ['L', 'G']:
            for probe_C in ['L', 'G']:
                for probe_D in ['L', 'G']:
                    Sijkl_dict[probe_A, probe_B, probe_C, probe_D] = \
                        Sijkl[probe_lookup[probe_A]['start']:probe_lookup[probe_A]['stop'],
                        probe_lookup[probe_B]['start']:probe_lookup[probe_B]['stop'],
                        probe_lookup[probe_C]['start']:probe_lookup[probe_C]['stop'],
                        probe_lookup[probe_D]['start']:probe_lookup[probe_D]['stop']]

    return Sijkl_dict


@njit
def build_3x2pt_dict(array_3x2pt):
    dict_3x2pt = {}
    if array_3x2pt.ndim == 5:
        dict_3x2pt['L', 'L'] = array_3x2pt[:, 0, 0, :, :]
        dict_3x2pt['L', 'G'] = array_3x2pt[:, 0, 1, :, :]
        dict_3x2pt['G', 'L'] = array_3x2pt[:, 1, 0, :, :]
        dict_3x2pt['G', 'G'] = array_3x2pt[:, 1, 1, :, :]
    elif array_3x2pt.ndim == 4:
        dict_3x2pt['L', 'L'] = array_3x2pt[0, 0, :, :]
        dict_3x2pt['L', 'G'] = array_3x2pt[0, 1, :, :]
        dict_3x2pt['G', 'L'] = array_3x2pt[1, 0, :, :]
        dict_3x2pt['G', 'G'] = array_3x2pt[1, 1, :, :]
    return dict_3x2pt


# ! to be deprecated
@njit
def cov_SS_3x2pt_10D_dict_nofsky_old(nbl, cl_3x2pt, Sijkl, zbins, response_3x2pt, probe_ordering):
    """Buil the 3x2pt covariance matrix using a dict for the response, the cls and Sijkl.
    Slightly slower (because of the use of dicts, I think) but much cleaner (no need for multiple if statements).
    """
    #
    # build and/or initialize the dictionaries
    Sijkl_dict = build_Sijkl_dict(Sijkl, zbins)
    cl_3x2pt_dict = build_3x2pt_dict(cl_3x2pt)
    response_3x2pt_dict = build_3x2pt_dict(response_3x2pt)
    cov_3x2pt_SS_10D = {}

    # compute the SS cov only for the relevant probe combinations
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            cov_3x2pt_SS_10D[A, B, C, D] = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
            for ell1 in range(nbl):
                for ell2 in range(nbl):
                    for i in range(zbins):
                        for j in range(zbins):
                            for k in range(zbins):
                                for l in range(zbins):
                                    cov_3x2pt_SS_10D[A, B, C, D][ell1, ell2, i, j, k, l] = \
                                        (response_3x2pt_dict[A, B][ell1, i, j] *
                                         response_3x2pt_dict[C, D][ell2, k, l] *
                                         cl_3x2pt_dict[A, B][ell1, i, j] *
                                         cl_3x2pt_dict[C, D][ell2, k, l] *
                                         Sijkl_dict[A, B, C, D][i, j, k, l])
            print('computing SSC in blocks: working probe combination', A, B, C, D)

    return cov_3x2pt_SS_10D


# ! to be deprecated
def cov_SS_3x2pt_10D_dict_old(nbl, cl_3x2pt, Sijkl, fsky, zbins, response_3x2pt, probe_ordering):
    """ this function exists only because numba does not let me perform this simple renormalization, and dividing
    each element by fsky is very inefficient. It's a very simple wrapper of covariance_SS_3x2pt_10D_dict_nofsky"""

    # compute covSSC
    cov_3x2pt_SS_10D = cov_SS_3x2pt_10D_dict_nofsky_old(nbl, cl_3x2pt, Sijkl, zbins, response_3x2pt, probe_ordering)

    # divide it by fsky
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            cov_3x2pt_SS_10D[A, B, C, D][...] /= fsky

    return cov_3x2pt_SS_10D


@njit
def cov_SSC_ALL(nbl, npairs_tot, ind, D_3x2pt, Sijkl, fsky, zbins, Rl):
    """The fastest routine to compute the SSC covariance matrix.
    """

    cov_ALL_SSC = np.zeros((nbl, nbl, npairs_tot, npairs_tot))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs_tot):
                for q in range(npairs_tot):
                    i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]
                    A, B, C, D = ind[p, 0], ind[p, 1], ind[q, 0], ind[q, 1]

                    # the shift is implemented by multiplying A, B, C, D by zbins: if lensing, probe == 0 and shift = 0
                    # if probe is GC, probe == 1 and shift = zbins. this does not hold if you switch probe indices!
                    cov_ALL_SSC[ell1, ell2, p, q] = (Rl[ell1, A, B, i, j] *
                                                     Rl[ell2, C, D, k, l] *
                                                     D_3x2pt[ell1, A, B, i, j] *
                                                     D_3x2pt[ell2, C, D, k, l] *
                                                     Sijkl[i + A * zbins, j + B * zbins, k + C * zbins, l + D * zbins])

    cov_ALL_SSC /= fsky
    return cov_ALL_SSC


# ! to be deprecated
@njit
def cov_SSC_ALL_dict(nbl, npairs_tot, ind, D_3x2pt, Sijkl, fsky, zbins, Rl):
    """Buil the 3x2pt covariance matrix using a dict for Sijkl. slightly slower (because of the use of dicts, I think)
    but cleaner (no need for multiple if statements, except to set the correct probes).
    Note that the ell1, ell2 slicing does not work! You can substitute only one of the for loops (in this case the one over ell1).
    A_str = probe A as string (e.g. 'L' for lensing)
    A_num = probe A as number (e.g. 0 for lensing)
    """

    Sijkl_dict = build_Sijkl_dict(Sijkl, zbins)
    print('xxxxx x xx  x x x x  x x x x  x x xvariable response not implemented!')

    cov_3x2pt_SSC = np.zeros((nbl, nbl, npairs_tot, npairs_tot))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs_tot):
                for q in range(npairs_tot):

                    # TODO do this with a dictionary!!
                    if ind[p, 0] == 0:
                        A_str = 'L'
                    elif ind[p, 0] == 1:
                        A_str = 'G'
                    if ind[p, 1] == 0:
                        B_str = 'L'
                    elif ind[p, 1] == 1:
                        B_str = 'G'
                    if ind[q, 0] == 0:
                        C_str = 'L'
                    elif ind[q, 0] == 1:
                        C_str = 'G'
                    if ind[q, 1] == 0:
                        D_str = 'L'
                    elif ind[q, 1] == 1:
                        D_str = 'G'

                    A_num, B_num, C_num, D_num = ind[p, 0], ind[p, 1], ind[q, 0], ind[q, 1]
                    i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]

                    cov_3x2pt_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                       D_3x2pt[ell1, A_num, B_num, i, j] *
                                                       D_3x2pt[ell2, C_num, D_num, k, l] *
                                                       Sijkl_dict[A_str, B_str, C_str, D_str][i, j, k, l])
    return cov_3x2pt_SSC / fsky


def cov_G_10D_dict(cl_dict, noise_dict, nbl, zbins, l_lin, delta_l, fsky, probe_ordering):
    """
    A universal 6D covmat function, which mixes the indices automatically. 
    This one works with dictionaries, in particular for the cls and noise arrays. 
    probe_ordering = ['L', 'L'] or ['G', 'G'] for the individual probes, and
    probe_ordering = [['L', 'L'], ['L', 'G'], ['G', 'G']] (or variations)
    for the 3x2pt case.
    Note that, adding together the different datavectors, cov_3x2pt_6D needs
    probe indices, becoming 10D (maybe a (nbl, nbl, 3*zbins, 3*zbins, 3*zbins, 3*zbins))
    shape would work? Anyway, much less convenient to work with.
    
    This version is faster, it is a wrapper function for covariance_6D_blocks,
    which makes use of jit
    """

    cov_10D_dict = {}
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            cov_10D_dict[A, B, C, D] = cov_GO_6D_blocks(
                cl_dict[A, C], cl_dict[B, D], cl_dict[A, D], cl_dict[B, C],
                noise_dict[A, C], noise_dict[B, D], noise_dict[A, D], noise_dict[B, C],
                nbl, zbins, l_lin, delta_l, fsky)
    return cov_10D_dict


def cov_SS_10D_dict(Cl_dict, Rl_dict, Sijkl_dict, nbl, zbins, fsky, probe_ordering):
    """
    A universal 6D covmat function, which mixes the indices automatically.
    This one works with dictionaries, in particular for the cls and noise arrays.
    probe_ordering = ['L', 'L'] or ['G', 'G'] for the individual probes, and
    probe_ordering = [['L', 'L'], ['L', 'G'], ['G', 'G']] (or variations)
    for the 3x2pt case.
    Note that, adding together the different datavectors, cov_3x2pt_6D needs
    probe indices, becoming 10D (maybe a (nbl, nbl, 3*zbins, 3*zbins, 3*zbins, 3*zbins))
    shape would work? Anyway, much less convenient to work with.

    This version is faster, it is a wrapper function for covariance_6D_blocks,
    which makes use of jit
    """

    cov_SS_10D_dict = {}
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            cov_SS_10D_dict[A, B, C, D] = cov_SS_6D_blocks(Rl_dict[A, B], Cl_dict[A, B], Rl_dict[C, D], Cl_dict[C, D],
                                                           Sijkl_dict[A, B, C, D], nbl, zbins, fsky)

    return cov_SS_10D_dict


# This function does mix the indices, but not automatically: it only indicates which ones to use and where
# It can be used for the individual blocks of the 3x2pt (unlike the one above),
# but it has to be called once for each block combination (see cov_blocks_LG_4D
# and cov_blocks_GL_4D)
# best used in combination with cov_10D_dictionary
@njit
def cov_GO_6D_blocks(C_AC, C_BD, C_AD, C_BC, N_AC, N_BD, N_AD, N_BC, nbl, zbins, l_lin, delta_l, fsky):
    cov_GO_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ell in range(nbl):
        for i in range(zbins):
            for j in range(zbins):
                for k in range(zbins):
                    for l in range(zbins):
                        cov_GO_6D[ell, ell, i, j, k, l] = \
                            ((C_AC[ell, i, k] + N_AC[i, k]) *
                             (C_BD[ell, j, l] + N_BD[j, l]) +
                             (C_AD[ell, i, l] + N_AD[i, l]) *
                             (C_BC[ell, j, k] + N_BC[j, k])) / \
                            ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])
    return cov_GO_6D


@njit
def cov_SS_6D_blocks(Rl_AB, Cl_AB, Rl_CD, Cl_CD, Sijkl_ABCD, nbl, zbins, fsky):
    """ experimental"""
    cov_SS_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for i in range(zbins):
                for j in range(zbins):
                    for k in range(zbins):
                        for l in range(zbins):
                            cov_SS_6D[ell1, ell2, i, j, k, l] = \
                                (Rl_AB[ell1, i, j] *
                                 Cl_AB[ell1, i, j] *
                                 Rl_CD[ell2, k, l] *
                                 Cl_CD[ell2, k, l] *
                                 Sijkl_ABCD[i, j, k, l])
    cov_SS_6D /= fsky
    return cov_SS_6D


def cov_3x2pt_dict_10D_to_4D(cov_3x2pt_dict_10D, probe_ordering, nbl, zbins, ind_copy, GL_or_LG):
    """
    Takes the cov_3x2pt_10D dictionary, reshapes each A, B, C, D block separately
    in 4D, then stacks the blocks in the right order to output cov_3x2pt_4D 
    (which is not a dictionary but a numpy array)

    probe_ordering: e.g. ['L', 'L'], ['G', 'L'], ['G', 'G']]
    """

    ind_copy = ind_copy.copy()  # just to ensure the input ind file is not changed

    # Check that the cross-correlation is coherent with the probe_ordering list
    # this is a weak check, since I'm assuming that GL or LG will be the second 
    # element of the datavector
    if GL_or_LG == 'GL':
        assert probe_ordering[1][0] == 'G' and probe_ordering[1][1] == 'L', \
            'probe_ordering[1] should be "GL", e.g. [LL, GL, GG]'
    elif GL_or_LG == 'LG':
        assert probe_ordering[1][0] == 'L' and probe_ordering[1][1] == 'G', \
            'probe_ordering[1] should be "LG", e.g. [LL, LG, GG]'

    # get npairs
    npairs_auto, npairs_cross, npairs_3x2pt = get_zpairs(zbins)

    # construct the ind dict
    ind_dict = {}
    ind_dict['L', 'L'] = ind_copy[:npairs_auto, :]
    ind_dict['G', 'G'] = ind_copy[(npairs_auto + npairs_cross):, :]
    if GL_or_LG == 'LG':
        ind_dict['L', 'G'] = ind_copy[npairs_auto:(npairs_auto + npairs_cross), :]
        ind_dict['G', 'L'] = ind_dict['L', 'G'].copy()  # copy and switch columns
        ind_dict['G', 'L'][:, [2, 3]] = ind_dict['G', 'L'][:, [3, 2]]
    elif GL_or_LG == 'GL':
        ind_dict['G', 'L'] = ind_copy[npairs_auto:(npairs_auto + npairs_cross), :]
        ind_dict['L', 'G'] = ind_dict['G', 'L'].copy()  # copy and switch columns
        ind_dict['L', 'G'][:, [2, 3]] = ind_dict['L', 'G'][:, [3, 2]]

    # construct the npairs dict 
    npairs_dict = {}
    npairs_dict['L', 'L'] = npairs_auto
    npairs_dict['L', 'G'] = npairs_cross
    npairs_dict['G', 'L'] = npairs_cross
    npairs_dict['G', 'G'] = npairs_auto

    # initialize the 4D dictionary and list of probe combinations
    cov_3x2pt_dict_4D = {}
    combinations = []

    # make each block 4D and store it with the right 'A', 'B', 'C, 'D' key 
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            combinations.append([A, B, C, D])
            cov_3x2pt_dict_4D[A, B, C, D] = cov_6D_to_4D_blocks(cov_3x2pt_dict_10D[A, B, C, D], nbl, npairs_dict[A, B],
                                                                npairs_dict[C, D], ind_dict[A, B], ind_dict[C, D])

    # take the correct combinations (stored in 'combinations') and construct
    # lists which will be converted to arrays
    row_1_list = [cov_3x2pt_dict_4D[A, B, C, D] for A, B, C, D in combinations[:3]]
    row_2_list = [cov_3x2pt_dict_4D[A, B, C, D] for A, B, C, D in combinations[3:6]]
    row_3_list = [cov_3x2pt_dict_4D[A, B, C, D] for A, B, C, D in combinations[6:9]]

    # concatenate the lists to make rows
    row_1 = np.concatenate(row_1_list, axis=3)
    row_2 = np.concatenate(row_2_list, axis=3)
    row_3 = np.concatenate(row_3_list, axis=3)

    # concatenate the rows to construct the final matrix
    cov_3x2pt_4D = np.concatenate((row_1, row_2, row_3), axis=2)

    return cov_3x2pt_4D


# ! to be deprecated
@njit
def symmetrize_ij(cov_6D, zbins):
    warnings.warn('THIS FUNCTION ONLY WORKS IF THE MATRIX TO SYMMETRIZE IS UPPER *OR* LOWER TRIANGULAR, NOT BOTH')
    # TODO thorough check?
    for i in range(zbins):
        for j in range(zbins):
            cov_6D[:, :, i, j, :, :] = cov_6D[:, :, j, i, :, :]
            cov_6D[:, :, :, :, i, j] = cov_6D[:, :, :, :, j, i]
    return cov_6D


# @njit
# ! this function is new - still to be thouroughly tested
def cov_4D_to_6D(cov_4D, nbl, zbins, probe, ind):
    """transform the cov from shape (nbl, nbl, npairs, npairs) 
    to (nbl, nbl, zbins, zbins, zbins, zbins). Not valid for 3x2pt, the total
    shape of the matrix is (nbl, nbl, zbins, zbins, zbins, zbins), not big 
    enough to store 3 probes. Use cov_4D functions or cov_6D as a dictionary
    instead,
    """

    assert probe in ['LL', 'GG', 'LG', 'GL'], 'probe must be "LL", "LG", "GL" or "GG". 3x2pt is not supported'

    npairs_auto, npairs_cross, npairs_tot = get_zpairs(zbins)
    if probe in ['LL', 'GG']:
        npairs = npairs_auto
    elif probe in ['GL', 'LG']:
        npairs = npairs_cross

    assert ind.shape[0] == npairs, 'ind.shape[0] != npairs: maybe you\'re passing the whole ind file ' \
                                   'instead of ind[:npairs, :] - or similia'

    # TODO use jit
    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ij in range(npairs):
        for kl in range(npairs):
            i, j, k, l = ind[ij, 2], ind[ij, 3], ind[kl, 2], ind[kl, 3]
            cov_6D[:, :, i, j, k, l] = cov_4D[:, :, ij, kl]

    # GL is not symmetric
    # ! this part makes this function very slow
    if probe in ['LL', 'GG']:
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                for i in range(zbins):
                    for j in range(zbins):
                        cov_6D[ell1, ell2, :, :, i, j] = symmetrize_2d_array(cov_6D[ell1, ell2, :, :, i, j])
                        cov_6D[ell1, ell2, i, j, :, :] = symmetrize_2d_array(cov_6D[ell1, ell2, i, j, :, :])

    return cov_6D


# @njit
def cov_6D_to_4D(cov_6D, nbl, zpairs, ind):
    """transform the cov from shape (nbl, nbl, zbins, zbins, zbins, zbins) 
    to (nbl, nbl, zpairs, zpairs)"""
    assert ind.shape[0] == zpairs, "ind.shape[0] != zpairs: maybe you're passing the whole ind file " \
                                   "instead of ind[:zpairs, :] - or similia"
    cov_4D = np.zeros((nbl, nbl, zpairs, zpairs))
    for ij in range(zpairs):
        for kl in range(zpairs):
            # rename for better readability
            i, j, k, l = ind[ij, -2], ind[ij, -1], ind[kl, -2], ind[kl, -1]
            cov_4D[:, :, ij, kl] = cov_6D[:, :, i, j, k, l]
    return cov_4D


@njit
def cov_6D_to_4D_blocks(cov_6D, nbl, npairs_AB, npairs_CD, ind_AB, ind_CD):
    """ reshapes the covariance even for the non-diagonal (hence, non-square) blocks needed to build the 3x2pt.
    use npairs_AB = npairs_CD and ind_AB = ind_CD for the normal routine (valid for auto-covariance 
    LL-LL, GG-GG, GL-GL and LG-LG). n_columns is used to determine whether the ind array has 2 or 4 columns
    (if it's given in the form of a dictionary or not)
    """
    assert ind_AB.shape[0] == npairs_AB, 'ind_AB.shape[0] != npairs_AB'
    assert ind_CD.shape[0] == npairs_CD, 'ind_CD.shape[0] != npairs_CD'

    # this is to ensure compatibility with both 4-columns and 2-columns ind arrays (dictionary)
    # the penultimante element is the first index, the last one the second index (see s - 1, s - 2 below)
    n_columns_AB = ind_AB.shape[1]  # of columns: this is to understand the format of the file
    n_columns_CD = ind_CD.shape[1]

    # check
    assert n_columns_AB == n_columns_CD, 'ind_AB and ind_CD must have the same number of columns'
    nc = n_columns_AB  # make the name shorter

    cov_4D = np.zeros((nbl, nbl, npairs_AB, npairs_CD))
    for ij in range(npairs_AB):
        for kl in range(npairs_CD):
            i, j, k, l = ind_AB[ij, nc - 2], ind_AB[ij, nc - 1], ind_CD[kl, nc - 2], ind_CD[kl, nc - 1]
            cov_4D[:, :, ij, kl] = cov_6D[:, :, i, j, k, l]
    return cov_4D


def return_combinations(A, B, C, D):
    print(f'C_{A}{C}, C_{B}{D}, C_{A}{D}, C_{B}{C}, N_{A}{C}, N_{B}{D}, N_{A}{D}, N_{B}{C}')


###########################
# @njit
def check_symmetric(array_2d, exact, rtol=1e-05):
    """
    :param a: 2d array
    :param exact: bool
    :param rtol: relative tolerance
    :return: bool, whether the array is symmetric or not
    """
    # """check if the matrix is symmetric, either exactly or within a tolerance
    # """
    assert type(exact) == bool, 'parameter "exact" must be either True or False'
    assert array_2d.ndim == 2, 'the array is not square'
    if exact:
        return np.array_equal(array_2d, array_2d.T)
    else:
        return np.allclose(array_2d, array_2d.T, rtol=rtol, atol=0)


@njit
# reshape from 3 to 4 dimensions
def array_3D_to_4D(cov_3D, nbl, npairs):
    print('XXX THIS FUNCTION ONLY WORKS FOR GAUSS-ONLY COVARIANCE')
    cov_4D = np.zeros((nbl, nbl, npairs, npairs))
    for ell in range(nbl):
        for p in range(npairs):
            for q in range(npairs):
                cov_4D[ell, ell, p, q] = cov_3D[ell, p, q]
    return cov_4D


# @njit
def cov_2D_to_4D(cov_2D, nbl, block_index='vincenzo'):
    """ new (more elegant) version of cov_2D_to_4D. Also works for 3x2pt. The order
    of the for loops does not affect the result!
    
    Ordeting convention:
    - whether to use ij or ell as the outermost index (determined by the ordering of the for loops)
      This is going to be the index of the blocks in the 2D covariance matrix.
    Sylvain uses block_index == 'zpair_wise', me and Vincenzo block_index == 'ell':
    I add this distinction in the "if" to make it clearer.

    Note: this reshaping does not change the number of elements, we just go from [nbl, nbl, zpairs, zpairs] to
    [nbl*zpairs, nbl*zpairs]; hence no symmetrization or other methods to "fill in" the missing elements in the
    higher-dimensional array are needed.
    """

    assert block_index in ['ell', 'vincenzo', 'C-style'] + ['ij', 'sylvain', 'F-style'], \
        'block_index must be "ell", "vincenzo", "C-style" or "ij", "sylvain", "F-style"'
    assert cov_2D.ndim == 2, 'the input covariance must be 2-dimensional'

    zpairs_AB = cov_2D.shape[0] // nbl
    zpairs_CD = cov_2D.shape[1] // nbl

    cov_4D = np.zeros((nbl, nbl, zpairs_AB, zpairs_CD))

    if block_index in ['ell', 'vincenzo', 'C-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_4D[l1, l2, ipair, jpair] = cov_2D[l1 * zpairs_AB + ipair, l2 * zpairs_CD + jpair]

    elif block_index in ['ij', 'sylvain', 'F-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_4D[l1, l2, ipair, jpair] = cov_2D[ipair * nbl + l1, jpair * nbl + l2]
    return cov_4D


@njit
def cov_4D_to_2D(cov_4D, block_index='vincenzo'):
    """ new (more elegant) version of cov_4D_to_2D. Also works for 3x2pt. The order
    of the for loops does not affect the result!

    Ordeting convention:
    - whether to use ij or ell as the outermost index (determined by the ordering of the for loops).
      This is going to be the index of the blocks in the 2D covariance matrix.
    Sylvain uses block_index == 'pair_wise', me and Vincenzo block_index == 'ell_wise':
    I add this distinction in the "if" to make it clearer.

    this function can also convert to 2D non-square blocks; this is needed to build the 3x2pt_2D according to CLOE's
    ordering (which is not actually Cloe's ordering...); it is sufficient to pass a zpairs_CD != zpairs_AB value
    (by default zpairs_CD == zpairs_AB). This is not necessary in the above function (unless you want to reshape the
    individual blocks) because also in the 3x2pt case I am reshaping a square matrix (of size [nbl*zpairs, nbl*zpairs])

    Note: this reshaping does not change the number of elements, we just go from [nbl, nbl, zpairs, zpairs] to
    [nbl*zpairs, nbl*zpairs]; hence no symmetrization or other methods to "fill in" the missing elements in the
    higher-dimensional array are needed.
    """

    assert block_index in ['ell', 'vincenzo', 'C-style'] + ['ij', 'sylvain', 'F-style'], \
        'block_index must be "ell", "vincenzo", "C-style" or "ij", "sylvain", "F-style"'

    assert cov_4D.ndim == 4, 'the input covariance must be 4-dimensional'
    assert cov_4D.shape[0] == cov_4D.shape[1], 'the first two axes of the input covariance must have the same size'
    # assert cov_4D.shape[2] == cov_4D.shape[3], 'the second two axes of the input covariance must have the same size'

    nbl = int(cov_4D.shape[0])
    zpairs_AB = int(cov_4D.shape[2])
    zpairs_CD = int(cov_4D.shape[3])

    cov_2D = np.zeros((nbl * zpairs_AB, nbl * zpairs_CD))

    if block_index in ['ell', 'vincenzo', 'C-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[l1 * zpairs_AB + ipair, l2 * zpairs_CD + jpair] = cov_4D[l1, l2, ipair, jpair]

    elif block_index in ['ij', 'sylvain', 'F-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[ipair * nbl + l1, jpair * nbl + l2] = cov_4D[l1, l2, ipair, jpair]
    return cov_2D


@njit
def cov_4D_to_2D_v0(cov_4D, nbl, zpairs_AB, zpairs_CD=None, block_index='vincenzo'):
    """ new (more elegant) version of cov_4D_to_2D. Also works for 3x2pt. The order
    of the for loops does not affect the result!

    Ordeting convention:
    - whether to use ij or ell as the outermost index (determined by the ordering of the for loops).
      This is going to be the index of the blocks in the 2D covariance matrix.
    Sylvain uses block_index == 'pair_wise', me and Vincenzo block_index == 'ell_wise':
    I add this distinction in the "if" to make it clearer.

    this function can also convert to 2D non-square blocks; this is needed to build the 3x2pt_2D according to CLOE's
    ordering (which is not actually Cloe's ordering...); it is sufficient to pass a zpairs_CD != zpairs_AB value
    (by default zpairs_CD == zpairs_AB). This is not necessary in the above function (unless you want to reshape the
    individual blocks) because also in the 3x2pt case I am reshaping a square matrix (of size [nbl*zpairs, nbl*zpairs])

    Note: this reshaping does not change the number of elements, we just go from [nbl, nbl, zpairs, zpairs] to
    [nbl*zpairs, nbl*zpairs]; hence no symmetrization or other methods to "fill in" the missing elements in the
    higher-dimensional array are needed.
    """

    assert block_index in ['ell', 'vincenzo', 'C-style'] + ['ij', 'sylvain', 'F-style'], \
        'block_index must be "ell", "vincenzo", "C-style" or "ij", "sylvain", "F-style"'

    assert cov_4D.ndim == 4, 'the input covariance must be 4-dimensional'
    assert cov_4D.shape[0] == cov_4D.shape[1], 'the first two axes of the input covariance must have the same size'
    assert cov_4D.shape[2] == cov_4D.shape[3], 'the second two axes of the input covariance must have the same size'

    # if not passed, zpairs_CD must be equal to zpairs_AB
    if zpairs_CD is None:
        zpairs_CD = zpairs_AB

    if zpairs_AB != zpairs_CD:
        print('warning: zpairs_AB != zpairs_CD, the output covariance will be non-square')

    cov_2D = np.zeros((nbl * zpairs_AB, nbl * zpairs_CD))

    if block_index in ['ell', 'vincenzo', 'C-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[l1 * zpairs_AB + ipair, l2 * zpairs_CD + jpair] = cov_4D[l1, l2, ipair, jpair]

    elif block_index in ['ij', 'sylvain', 'F-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[ipair * nbl + l1, jpair * nbl + l2] = cov_4D[l1, l2, ipair, jpair]
    return cov_2D


# @njit
def cov_4D_to_2DCLOE_3x2pt(cov_4D, nbl, zbins, block_index='vincenzo'):
    """
    Reshape according to the "multi-diagonal", non-square blocks 2D_CLOE ordering. Note that this is only necessary for
    the 3x2pt probe.
    TODO the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it
    TODO will work both for LG and GL)
    """

    print("the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it"
          " will work both for LG and GL) ")

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    lim_1 = zpairs_auto
    lim_2 = zpairs_cross + zpairs_auto
    lim_3 = zpairs_3x2pt

    # note: I'm writing cov_LG, but there should be no issue with GL; after all, this function is not using the ind file
    cov_LL_LL = cov_4D_to_2D(cov_4D[:, :, :lim_1, :lim_1], block_index)
    cov_LL_LG = cov_4D_to_2D(cov_4D[:, :, :lim_1, lim_1:lim_2], block_index)
    cov_LL_GG = cov_4D_to_2D(cov_4D[:, :, :lim_1, lim_2:lim_3], block_index)

    cov_LG_LL = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, :lim_1], block_index)
    cov_LG_LG = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, lim_1:lim_2], block_index)
    cov_LG_GG = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, lim_2:lim_3], block_index)

    cov_GG_LL = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, :lim_1], block_index)
    cov_GG_LG = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, lim_1:lim_2], block_index)
    cov_GG_GG = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, lim_2:lim_3], block_index)

    # make long rows and stack together
    row_1 = np.hstack((cov_LL_LL, cov_LL_LG, cov_LL_GG))
    row_2 = np.hstack((cov_LG_LL, cov_LG_LG, cov_LG_GG))
    row_3 = np.hstack((cov_GG_LL, cov_GG_LG, cov_GG_GG))

    array_2D = np.vstack((row_1, row_2, row_3))

    return array_2D


# @njit
def cov_2DCLOE_to_4D_3x2pt(cov_2D, nbl, zbins, block_index='vincenzo'):
    """
    Reshape according to the "multi-diagonal", non-square blocks 2D_CLOE ordering. Note that this is only necessary for
    the 3x2pt probe.
    TODO the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it
    TODO will work both for LG and GL)
    """

    print("the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it"
          " will work both for LG and GL) ")

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    # now I'm reshaping the full block diagonal matrix, not just the sub-blocks (cov_2D_to_4D works for both cases)
    lim_1 = zpairs_auto * nbl
    lim_2 = (zpairs_cross + zpairs_auto) * nbl
    lim_3 = zpairs_3x2pt * nbl

    # note: I'm writing cov_LG, but there should be no issue with GL; after all, this function is not using the ind file
    cov_LL_LL = cov_2D_to_4D(cov_2D[:lim_1, :lim_1], nbl, block_index)
    cov_LL_LG = cov_2D_to_4D(cov_2D[:lim_1, lim_1:lim_2], nbl, block_index)
    cov_LL_GG = cov_2D_to_4D(cov_2D[:lim_1, lim_2:lim_3], nbl, block_index)

    cov_LG_LL = cov_2D_to_4D(cov_2D[lim_1:lim_2, :lim_1], nbl, block_index)
    cov_LG_LG = cov_2D_to_4D(cov_2D[lim_1:lim_2, lim_1:lim_2], nbl, block_index)
    cov_LG_GG = cov_2D_to_4D(cov_2D[lim_1:lim_2, lim_2:lim_3], nbl, block_index)

    cov_GG_LL = cov_2D_to_4D(cov_2D[lim_2:lim_3, :lim_1], nbl, block_index)
    cov_GG_LG = cov_2D_to_4D(cov_2D[lim_2:lim_3, lim_1:lim_2], nbl, block_index)
    cov_GG_GG = cov_2D_to_4D(cov_2D[lim_2:lim_3, lim_2:lim_3], nbl, block_index)

    # here it is a little more difficult to visualize the stacking, but the probes are concatenated
    # along the 2 zpair_3x2pt-long axes
    cov_4D = np.zeros((nbl, nbl, zpairs_3x2pt, zpairs_3x2pt))

    zlim_1 = zpairs_auto
    zlim_2 = zpairs_cross + zpairs_auto
    zlim_3 = zpairs_3x2pt

    cov_4D[:, :, :zlim_1, :zlim_1] = cov_LL_LL
    cov_4D[:, :, :zlim_1, zlim_1:zlim_2] = cov_LL_LG
    cov_4D[:, :, :zlim_1, zlim_2:zlim_3] = cov_LL_GG

    cov_4D[:, :, zlim_1:zlim_2, :zlim_1] = cov_LG_LL
    cov_4D[:, :, zlim_1:zlim_2, zlim_1:zlim_2] = cov_LG_LG
    cov_4D[:, :, zlim_1:zlim_2, zlim_2:zlim_3] = cov_LG_GG

    cov_4D[:, :, zlim_2:zlim_3, :zlim_1] = cov_GG_LL
    cov_4D[:, :, zlim_2:zlim_3, zlim_1:zlim_2] = cov_GG_LG
    cov_4D[:, :, zlim_2:zlim_3, zlim_2:zlim_3] = cov_GG_GG

    return cov_4D


def cov_4D_to_2DCLOE_3x2pt_bu(cov_4D, nbl, zbins, block_index='vincenzo'):
    """
    Reshape according to the "multi-diagonal", non-square blocks 2D_CLOE ordering. Note that this is only necessary for
    the 3x2pt probe.
    TODO the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it
    TODO will work both for LG and GL)
    """

    print("the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it"
          " will work both for LG and GL) ")

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    lim_1 = zpairs_auto
    lim_2 = zpairs_cross + zpairs_auto
    lim_3 = zpairs_3x2pt

    # note: I'm writing cov_LG, but there should be no issue with GL; after all, this function is not using the ind file
    cov_LL_LL = cov_4D_to_2D(cov_4D[:, :, :lim_1, :lim_1], nbl, zpairs_auto, zpairs_auto, block_index)
    cov_LL_LG = cov_4D_to_2D(cov_4D[:, :, :lim_1, lim_1:lim_2], nbl, zpairs_auto, zpairs_cross, block_index)
    cov_LL_GG = cov_4D_to_2D(cov_4D[:, :, :lim_1, lim_2:lim_3], nbl, zpairs_auto, zpairs_auto, block_index)

    cov_LG_LL = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, :lim_1], nbl, zpairs_cross, zpairs_auto, block_index)
    cov_LG_LG = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, lim_1:lim_2], nbl, zpairs_cross, zpairs_cross, block_index)
    cov_LG_GG = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, lim_2:lim_3], nbl, zpairs_cross, zpairs_auto, block_index)

    cov_GG_LL = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, :lim_1], nbl, zpairs_auto, zpairs_auto, block_index)
    cov_GG_LG = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, lim_1:lim_2], nbl, zpairs_auto, zpairs_cross, block_index)
    cov_GG_GG = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, lim_2:lim_3], nbl, zpairs_auto, zpairs_auto, block_index)

    # make long rows and stack together
    row_1 = np.hstack((cov_LL_LL, cov_LL_LG, cov_LL_GG))
    row_2 = np.hstack((cov_LG_LL, cov_LG_LG, cov_LG_GG))
    row_3 = np.hstack((cov_GG_LL, cov_GG_LG, cov_GG_GG))

    array_2D = np.vstack((row_1, row_2, row_3))

    return array_2D


def correlation_from_covariance(covariance):
    """ not thoroughly tested. Taken from 
    https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    does NOT work with 3x2pt
    """
    if covariance.shape[0] > 2000:
        print("this function doesn't work for 3x2pt")

    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def cov2corr(cov):
    """
    Convert a covariance matrix to a correlation matrix

    Args:
    cov (numpy.ndarray): A 2D covariance matrix

    Returns:
    numpy.ndarray: The corresponding 2D correlation matrix
    """
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    return corr


# compute Sylvain's deltas
def delta_l_Sylvain(nbl, ell):
    delta_l = np.zeros(nbl)
    for l in range(1, nbl):
        delta_l[l] = ell[l] - ell[l - 1]
    delta_l[0] = delta_l[1]
    return delta_l


def Recast_Sijkl_1xauto(Sijkl, zbins):
    npairs_auto = (zbins * (zbins + 1)) // 2
    pairs_auto = np.zeros((2, npairs_auto), dtype=int)
    count = 0
    for ibin in range(zbins):
        for jbin in range(ibin, zbins):
            pairs_auto[0, count] = ibin
            pairs_auto[1, count] = jbin
            count += 1
    Sijkl_recast = np.zeros((npairs_auto, npairs_auto))
    for ipair in range(npairs_auto):
        ibin = pairs_auto[0, ipair]
        jbin = pairs_auto[1, ipair]
        for jpair in range(npairs_auto):
            kbin = pairs_auto[0, jpair]
            lbin = pairs_auto[1, jpair]
            Sijkl_recast[ipair, jpair] = Sijkl[ibin, jbin, kbin, lbin]
    return [Sijkl_recast, npairs_auto, pairs_auto]


def Recast_Sijkl_3x2pt(Sijkl, nzbins):
    npairs_auto = (nzbins * (nzbins + 1)) // 2
    npairs_full = nzbins * nzbins + 2 * npairs_auto
    pairs_full = np.zeros((2, npairs_full), dtype=int)
    count = 0
    for ibin in range(nzbins):
        for jbin in range(ibin, nzbins):
            pairs_full[0, count] = ibin
            pairs_full[1, count] = jbin
            count += 1
    for ibin in range(nzbins, nzbins * 2):
        for jbin in range(nzbins):
            pairs_full[0, count] = ibin
            pairs_full[1, count] = jbin
            count += 1
    for ibin in range(nzbins, 2 * nzbins):
        for jbin in range(ibin, 2 * nzbins):
            pairs_full[0, count] = ibin
            pairs_full[1, count] = jbin
            count += 1
    Sijkl_recast = np.zeros((npairs_full, npairs_full))
    for ipair in range(npairs_full):
        ibin = pairs_full[0, ipair]
        jbin = pairs_full[1, ipair]
        for jpair in range(npairs_full):
            kbin = pairs_full[0, jpair]
            lbin = pairs_full[1, jpair]
            Sijkl_recast[ipair, jpair] = Sijkl[ibin, jbin, kbin, lbin]
    return [Sijkl_recast, npairs_full, pairs_full]


## build the noise matrices ##
def build_noise(zbins, nProbes, sigma_eps2, ng, EP_or_ED='EP'):
    """
    function to build the noise power spectra.
    ng = number of galaxies per arcmin^2 (constant, = 30 in IST:F 2020)
    n_bar = # of gal per bin
    """
    conversion_factor = 11818102.860035626  # deg to arcmin^2

    # if ng is a number, n_bar will be ng/zbins and the bins have to be equipopulated
    if type(ng) == int or type(ng) == float:
        assert ng > 0, 'ng should be positive'
        assert EP_or_ED == 'EP', 'if ng is a scalar (not a vector), the bins should be equipopulated'
        assert ng > 20, 'ng should roughly be > 20 (this check is meant to make sure that ng is the cumulative galaxy ' \
                        'density, not the galaxy density in each bin)'
        n_bar = ng / zbins * conversion_factor

    # if ng is an array, n_bar == ng (this is a slight minomer, since ng is the cumulative galaxy density, while
    # n_bar the galaxy density in each bin). In this case, if the bins are quipopulated, the n_bar array should
    # have all entries almost identical.
    elif type(ng) == np.ndarray:
        assert np.all(ng > 0), 'ng should be positive'
        assert np.sum(ng) > 20, 'ng should roughly be > 20'
        if EP_or_ED == 'EP':
            assert np.allclose(np.ones(ng) * ng[0], ng), 'if ng is a vector and the bins are equipopulated, ' \
                                                         'the value in each bin should be the same (or very similar)'
        n_bar = ng * conversion_factor

    else:
        raise ValueError('ng must be an int, float or numpy.ndarray')

    # create and fill N
    N = np.zeros((nProbes, nProbes, zbins, zbins))
    np.fill_diagonal(N[0, 0, :, :], sigma_eps2 / n_bar)
    np.fill_diagonal(N[1, 1, :, :], 1 / n_bar)
    N[0, 1, :, :] = 0
    N[1, 0, :, :] = 0
    return N


def my_exit():
    print('\nquitting script with sys.exit()')
    sys.exit()


########################### SYLVAINS FUNCTIONS ################################
@njit
def cov_4D_to_2D_sylvains_ord(cov_4D, nbl, npairs):
    """Reshape from 2D to 4D using Sylvain's ordering"""
    cov_2D = np.zeros((nbl * npairs, nbl * npairs))
    for ipair in range(npairs):
        for jpair in range(npairs):
            for l1 in range(nbl):
                for l2 in range(nbl):
                    cov_2D[ipair * nbl + l1, jpair * nbl + l2] = cov_4D[l1, l2, ipair, jpair]
    return cov_2D


@njit
def cov_2D_to_4D_sylvains_ord(cov_2D, nbl, npairs):
    """Reshape from 4D to 2D using Sylvain's ordering"""
    cov_4D = np.zeros((nbl, nbl, npairs, npairs))
    for ipair in range(npairs):
        for jpair in range(npairs):
            for l1 in range(nbl):
                for l2 in range(nbl):
                    cov_4D[l1, l2, ipair, jpair] = cov_2D[ipair * nbl + l1, jpair * nbl + l2]
    return cov_4D


def Cl_3D_to_1D(Cl_3D, nbl, npairs, ind, block_index='ij'):
    """This flattens the Cl_3D to 1D. Two ordeting conventions are used:
    - whether to use ij or ell as the outermost index (determined by the ordering of the for loops).
      This is going to be the index of the blocks in the 2D covariance matrix.
    - which ind file to use
    Sylvain uses block_index == 'pair_wise', me and Vincenzo block_index == 'ell_wise':
    I add this distinction in the "if" to make it clearer.
    """

    Cl_1D = np.zeros((nbl * npairs))

    k = 0
    if block_index == 'ij' or block_index == 'sylvain':
        for ij in range(npairs):  # ATTENTION TO THE LOOP ORDERING!!
            for ell in range(nbl):
                Cl_1D[k] = Cl_3D[ell, ind[ij, 0], ind[ij, 1]]
                k += 1

    elif block_index == 'ell' or block_index == 'vincenzo':
        for ell in range(nbl):
            for ij in range(npairs):  # ATTENTION TO THE LOOP ORDERING!!
                Cl_1D[k] = Cl_3D[ell, ind[ij, 0], ind[ij, 1]]
                k += 1

    return Cl_1D


########################### OLD FUNCTIONS ##############################################################################
########################################################################################################################

@njit
def cov_SSC_ALL_old_improved(nbl, npairs_tot, ind, D_3x2pt, Sijkl, fsky, zbins, Rl):
    """The fastest routine to compute the SSC covariance matrix.
    Implements the new shift, which is much better (no ifs!!!)
    Superseeded by passing Rl as an array (fill with the same values, in case of a constant probe response)
    """

    cov_ALL_SSC = np.zeros((nbl, nbl, npairs_tot, npairs_tot))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs_tot):
                for q in range(npairs_tot):
                    i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]
                    A, B, C, D = ind[p, 0], ind[p, 1], ind[q, 0], ind[q, 1]

                    # the shift is implemented by multiplying A, B, C, D by zbins: if lensing, probe == 0 and shift = 0
                    # if probe is GC, probe == 1 and shift = zbins. this does not hold if you switch probe indices!
                    cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                     D_3x2pt[ell1, A, B, i, j] *
                                                     D_3x2pt[ell2, C, D, k, l] *
                                                     Sijkl[i + A * zbins, j + B * zbins, k + C * zbins, l + D * zbins])

    cov_ALL_SSC /= fsky
    return cov_ALL_SSC


@njit
def cov_SSC_ALL_old(nbl, npairs_tot, ind, D_ALL, Sijkl, fsky, zbins, Rl):
    """Not the most elegant, but fast!
    """

    cov_ALL_SSC = np.zeros((nbl, nbl, npairs_tot, npairs_tot))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs_tot):
                for q in range(npairs_tot):

                    # LL_LL
                    if ind[p, 0] == 0 and ind[p, 1] == 0 and ind[q, 0] == 0 and ind[q, 1] == 0:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]])
                    # LL_GL
                    elif ind[p, 0] == 0 and ind[p, 1] == 0 and ind[q, 0] == 1 and ind[q, 1] == 0:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2], ind[p, 3], ind[q, 2] + zbins, ind[q, 3]])
                    # LL_GG
                    elif ind[p, 0] == 0 and ind[p, 1] == 0 and ind[q, 0] == 1 and ind[q, 1] == 1:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2], ind[p, 3], ind[q, 2] + zbins, ind[
                                                             q, 3] + zbins])

                    # GL_LL
                    elif ind[p, 0] == 1 and ind[p, 1] == 0 and ind[q, 0] == 0 and ind[q, 1] == 0:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2] + zbins, ind[p, 3], ind[q, 2], ind[q, 3]])
                    # GL_GL
                    elif ind[p, 0] == 1 and ind[p, 1] == 0 and ind[q, 0] == 1 and ind[q, 1] == 0:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2] + zbins, ind[p, 3], ind[q, 2] + zbins, ind[
                                                             q, 3]])
                    # GL_GG
                    elif ind[p, 0] == 1 and ind[p, 1] == 0 and ind[q, 0] == 1 and ind[q, 1] == 1:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2] + zbins, ind[p, 3], ind[q, 2] + zbins, ind[
                                                             q, 3] + zbins])

                    # GG_LL
                    elif ind[p, 0] == 1 and ind[p, 1] == 1 and ind[q, 0] == 0 and ind[q, 1] == 0:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2] + zbins, ind[p, 3] + zbins, ind[q, 2], ind[
                                                             q, 3]])
                    # GG_GL
                    elif ind[p, 0] == 1 and ind[p, 1] == 1 and ind[q, 0] == 1 and ind[q, 1] == 0:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2] + zbins, ind[p, 3] + zbins, ind[q, 2] + zbins,
                                                         ind[q, 3]])
                    # GG_GG
                    elif ind[p, 0] == 1 and ind[p, 1] == 1 and ind[q, 0] == 1 and ind[q, 1] == 1:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2] + zbins, ind[p, 3] + zbins, ind[q, 2] + zbins,
                                                               ind[q, 3] + zbins])
    cov_ALL_SSC /= fsky
    return cov_ALL_SSC


# @njit # not usable with dictionaries
def covariance_6D_dictionary_slow(cl_dict, noise_dict, nbl, zbins, l_lin, delta_l, fsky, probe_ordering):
    """
    A universal 6D covmat function, which mixes the indices automatically. 
    This one works with dictionaries, in particular for the cls and noise arrays. 
    probe_ordering = ['L', 'L'] or ['G', 'G'] for the individual probes, and
    probe_ordering = [['L', 'L'], ['L', 'G'], ['G', 'G']] (or variations)
    for the 3x2pt case.
    Note that, adding together the different datavectors, cov_3x2pt_6D needs
    probe indices, becoming 10D (maybe a (nbl, nbl, 3*zbins, 3*zbins, 3*zbins, 3*zbins))
    shape would work? Anyway, much less convenient to work with.
    
    this version is slower (but easier to read), it uses dictionaries directly 
    and cannot make use of numba jit
    """

    cov_10D_dict = {}
    cov_6D_arr = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            # this was the error: the array has to be initialized at every probe iteration!!!!
            cov_6D_arr = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
            for ell in range(nbl):
                for i in range(zbins):
                    for j in range(zbins):
                        for k in range(zbins):
                            for l in range(zbins):
                                cov_6D_arr[ell, ell, i, j, k, l] = \
                                    ((cl_dict[A, C][ell, i, k] + noise_dict[A, C][i, k]) *
                                     (cl_dict[B, D][ell, j, l] + noise_dict[B, D][j, l]) +
                                     (cl_dict[A, D][ell, i, l] + noise_dict[A, D][i, l]) *
                                     (cl_dict[B, C][ell, j, k] + noise_dict[B, C][j, k])) / \
                                    ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])
            cov_10D_dict[A, B, C, D] = cov_6D_arr

    return cov_10D_dict


# XXX these 2 are not fit for 3x2pt!!
# FIXME this function does not mix the A and B indices, is only fit for LL and GG
@njit
def covariance_6D(nbl, zbins, npairs, Cij, noise, l_lin, delta_l, fsky, ind, probe):
    print('this function is deprecated, use covariance_6D_dictionary instead')

    # some checks
    assert probe == "LL" or probe == "GG", 'probe must be LL or GG, this function cannot compute 3x2pt at the moment'
    if probe == "LL":
        probe_A = 0;
        probe_B = 0
    elif probe == "GG":
        probe_A = 1;
        probe_B = 1

    # create covariance array
    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ell in range(nbl):
        for i in range(zbins):
            for j in range(zbins):
                for k in range(zbins):
                    for l in range(zbins):
                        cov_6D[ell, ell, i, j, k, l] = \
                            ((Cij[ell, i, k] + noise[probe_A, probe_B, i, k]) *
                             (Cij[ell, j, l] + noise[probe_A, probe_B, j, l]) +
                             (Cij[ell, i, l] + noise[probe_A, probe_B, i, l]) *
                             (Cij[ell, j, k] + noise[probe_A, probe_B, j, k])) / \
                            ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])

    return cov_6D


# the following 2 are deprecated in favour of and cov_3x2pt_dict_10D_to_4D
def cov_blocks_GL_4D(D_ALL, N, nbl, zbins, l_lin_XC, delta_l_XC, fsky, ind, npairs, npairs_asimm):
    """
    computes the 3x2pt covariance in 6 blocks of 6D, then reshapes each block 
    individually to 4D and stacks everything into cov_3x2pt_4D. This one is specifically 
    made for the probe ordering (LL, GL, GG)
    """

    print('this function is deprecated, use covariance_6D_dictionary and cov_3x2pt_dict_10D_to_4D instead')

    C_LL = D_ALL[:, 0, 0, :, :]
    C_GG = D_ALL[:, 1, 1, :, :]
    C_LG = D_ALL[:, 0, 1, :, :]  # I'm renaming, should be correct, XXX BUG ALERT
    C_GL = D_ALL[:, 1, 0, :, :]

    # noise
    N_LL = N[0, 0, :, :]
    N_GG = N[1, 1, :, :]
    N_LG = N[0, 1, :, :]
    N_GL = N[1, 0, :, :]

    print('attention: there may be an issue with the ind array: \n ind[55:155, :] may actually be ind_LG, not ind_GL')
    print('THIS FUNCTION HAS TO BE FINISHED')

    ind_LL = ind[:55, :]
    ind_GG = ind[:55, :]
    ind_GL = ind[55:155, :]  # ind_GL????? XXX BUG ALERT
    ind_LG = np.copy(ind_GL)
    ind_LG[:, [2, 3]] = ind_LG[:, [3, 2]]

    cov_LL_LL_6D = cov_GO_6D_blocks(C_LL, C_LL, C_LL, C_LL, N_LL, N_LL, N_LL, N_LL, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)
    cov_LL_GL_6D = cov_GO_6D_blocks(C_LG, C_LL, C_LL, C_LG, N_LG, N_LL, N_LL, N_LG, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)
    cov_LL_GG_6D = cov_GO_6D_blocks(C_LG, C_LG, C_LG, C_LG, N_LG, N_LG, N_LG, N_LG, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)

    cov_GL_LL_6D = cov_GO_6D_blocks(C_GL, C_LL, C_GL, C_LL, N_GL, N_LL, N_GL, N_LL, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)
    cov_GL_GL_6D = cov_GO_6D_blocks(C_GG, C_LL, C_GL, C_LG, N_GG, N_LL, N_GL, N_LG, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)
    cov_GL_GG_6D = cov_GO_6D_blocks(C_GG, C_LG, C_GG, C_LG, N_GG, N_LG, N_GG, N_LG, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)

    cov_GG_LL_6D = cov_GO_6D_blocks(C_GL, C_GL, C_GL, C_GL, N_GL, N_GL, N_GL, N_GL, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)
    cov_GG_GL_6D = cov_GO_6D_blocks(C_GG, C_GL, C_GL, C_GG, N_GG, N_GL, N_GL, N_GG, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)
    cov_GG_GG_6D = cov_GO_6D_blocks(C_GG, C_GG, C_GG, C_GG, N_GG, N_GG, N_GG, N_GG, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)

    # 6D to 3D:
    cov_LL_LL_4D = cov_6D_to_4D_blocks(cov_LL_LL_6D, nbl, npairs, npairs, ind_LL, ind_LL)
    cov_LL_GL_4D = cov_6D_to_4D_blocks(cov_LL_GL_6D, nbl, npairs, npairs_asimm, ind_LL, ind_GL)  # ! it's ind_GL!
    cov_LL_GG_4D = cov_6D_to_4D_blocks(cov_LL_GG_6D, nbl, npairs, npairs, ind_LL, ind_GG)

    cov_GL_LL_4D = cov_6D_to_4D_blocks(cov_GL_LL_6D, nbl, npairs_asimm, npairs, ind_GL, ind_LL)
    cov_GL_GL_4D = cov_6D_to_4D_blocks(cov_GL_GL_6D, nbl, npairs_asimm, npairs_asimm, ind_GL, ind_GL)  # ! it's ind_GL!
    cov_GL_GG_4D = cov_6D_to_4D_blocks(cov_GL_GG_6D, nbl, npairs_asimm, npairs, ind_GL, ind_GG)

    cov_GG_LL_4D = cov_6D_to_4D_blocks(cov_GG_LL_6D, nbl, npairs, npairs, ind_GG, ind_LL)
    cov_GG_GL_4D = cov_6D_to_4D_blocks(cov_GG_GL_6D, nbl, npairs, npairs_asimm, ind_GG, ind_GL)  # ! it's ind_GL!
    cov_GG_GG_4D = cov_6D_to_4D_blocks(cov_GG_GG_6D, nbl, npairs, npairs, ind_GG, ind_GG)

    # put the matrix together
    row_1 = np.concatenate((cov_LL_LL_4D, cov_LL_GL_4D, cov_LL_GG_4D), axis=3)
    row_2 = np.concatenate((cov_GL_LL_4D, cov_GL_GL_4D, cov_GL_GG_4D), axis=3)
    row_3 = np.concatenate((cov_GG_LL_4D, cov_GG_GL_4D, cov_GG_GG_4D), axis=3)
    cov_4D_GL = np.concatenate((row_1, row_2, row_3), axis=2)

    return cov_4D_GL


def cov_blocks_LG_4D(D_ALL, N, nbl, zbins, l_lin_XC, delta_l_XC, fsky, ind, npairs, npairs_asimm):
    """
    computes the 3x2pt covariance in 6 blocks of 6D, then reshapes each block 
    individually to 4D and stacks everything into cov_3x2pt_4D. This one is specifically 
    made for the probe ordering (LL, LG, GG)
    """

    print('this function is deprecated, use covariance_6D_dictionary and cov_3x2pt_dict_10D_to_4D instead')

    C_LL = D_ALL[:, 0, 0, :, :]
    C_GG = D_ALL[:, 1, 1, :, :]
    C_LG = D_ALL[:, 0, 1, :, :]  # I'm renaming, should be correct, XXX BUG ALERT
    C_GL = D_ALL[:, 1, 0, :, :]

    # noise
    N_LL = N[0, 0, :, :]
    N_GG = N[1, 1, :, :]
    N_LG = N[0, 1, :, :]
    N_GL = N[1, 0, :, :]

    print('attention: there may be an issue with the ind array: \n ind[55:155, :] may actually be ind_LG, not ind_GL')
    print('THIS FUNCTION HAS TO BE FINISHED')
    ind_LL = ind[:55, :]
    ind_GG = ind[:55, :]
    ind_GL = ind[55:155, :]  # ind_GL????? XXX BUG ALERT
    ind_LG = np.copy(ind_GL)
    ind_LG[:, [2, 3]] = ind_LG[:, [3, 2]]

    # def cov_blocks_LG_6D(D_ALL, N, nbl, zbins, l_lin, delta_l, fsky):
    cov_LL_LL_6D = cov_GO_6D_blocks(C_LL, C_LL, C_LL, C_LL, N_LL, N_LL, N_LL, N_LL, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)
    cov_LL_LG_6D = cov_GO_6D_blocks(C_LL, C_LG, C_LG, C_LL, N_LL, N_LG, N_LG, N_LL, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)
    cov_LL_GG_6D = cov_GO_6D_blocks(C_LG, C_LG, C_LG, C_LG, N_LG, N_LG, N_LG, N_LG, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)

    cov_LG_LL_6D = cov_GO_6D_blocks(C_LL, C_GL, C_LL, C_GL, N_LL, N_GL, N_LL, N_GL, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)
    cov_LG_LG_6D = cov_GO_6D_blocks(C_LL, C_GG, C_LG, C_GL, N_LL, N_GG, N_LG, N_GL, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)
    cov_LG_GG_6D = cov_GO_6D_blocks(C_LG, C_GG, C_LG, C_GG, N_LG, N_GG, N_LG, N_GG, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)

    cov_GG_LL_6D = cov_GO_6D_blocks(C_GL, C_GL, C_GL, C_GL, N_GL, N_GL, N_GL, N_GL, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)
    cov_GG_LG_6D = cov_GO_6D_blocks(C_GL, C_GG, C_GG, C_GL, N_GL, N_GG, N_GG, N_GL, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)
    cov_GG_GG_6D = cov_GO_6D_blocks(C_GG, C_GG, C_GG, C_GG, N_GG, N_GG, N_GG, N_GG, nbl, zbins, l_lin_XC,
                                    delta_l_XC, fsky)

    # 6D to 3D:
    cov_LL_LL_4D = cov_6D_to_4D_blocks(cov_LL_LL_6D, nbl, npairs, npairs, ind_LL, ind_LL)
    cov_LL_LG_4D = cov_6D_to_4D_blocks(cov_LL_LG_6D, nbl, npairs, npairs_asimm, ind_LL,
                                       ind_GL)  # XXXXXXXXXXXX it's ind_GL!!!!!
    cov_LL_GG_4D = cov_6D_to_4D_blocks(cov_LL_GG_6D, nbl, npairs, npairs, ind_LL, ind_GG)

    cov_LG_LL_4D = cov_6D_to_4D_blocks(cov_LG_LL_6D, nbl, npairs_asimm, npairs, ind_GL, ind_LL)
    cov_LG_LG_4D = cov_6D_to_4D_blocks(cov_LG_LG_6D, nbl, npairs_asimm, npairs_asimm, ind_GL,
                                       ind_GL)  # XXXXXXXXXXXX it's ind_GL!!!!!
    cov_LG_GG_4D = cov_6D_to_4D_blocks(cov_LG_GG_6D, nbl, npairs_asimm, npairs, ind_GL, ind_GG)

    cov_GG_LL_4D = cov_6D_to_4D_blocks(cov_GG_LL_6D, nbl, npairs, npairs, ind_GG, ind_LL)
    cov_GG_LG_4D = cov_6D_to_4D_blocks(cov_GG_LG_6D, nbl, npairs, npairs_asimm, ind_GG,
                                       ind_GL)  # XXXXXXXXXXXX it's ind_GL!!!!!
    cov_GG_GG_4D = cov_6D_to_4D_blocks(cov_GG_GG_6D, nbl, npairs, npairs, ind_GG, ind_GG)

    # put the matrix together
    row_1 = np.concatenate((cov_LL_LL_4D, cov_LL_LG_4D, cov_LL_GG_4D), axis=3)
    row_2 = np.concatenate((cov_LG_LL_4D, cov_LG_LG_4D, cov_LG_GG_4D), axis=3)
    row_3 = np.concatenate((cov_GG_LL_4D, cov_GG_LG_4D, cov_GG_GG_4D), axis=3)
    cov_4D_LG = np.concatenate((row_1, row_2, row_3), axis=2)
    return cov_4D_LG


@njit
def cov_SSC_old(nbl, npairs, ind, Cij, Sijkl, fsky, probe, zbins, Rl):
    """old version. What changed in the new version is just the i, j, k, l 
    instead of the less readable ind[p, 2]... and so forth
    """

    assert probe in ['WL', 'WA', 'GC'], 'probe must be "WL", "WA" or "GC"'

    if probe == "WL" or probe == "WA":
        shift = 0
    elif probe == "GC":
        shift = zbins

    cov_SSC = np.zeros((nbl, nbl, npairs, npairs))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs):
                for q in range(npairs):
                    cov_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                 Cij[ell1, ind[p, 2], ind[p, 3]] *
                                                 Cij[ell2, ind[q, 2], ind[q, 3]] *
                                                 Sijkl[ind[p, 2] + shift, ind[p, 3] + shift,
                                                       ind[q, 2] + shift, ind[q, 3] + shift])
    cov_SSC /= fsky
    return cov_SSC


@njit
def cov_4D_to_2D_old(cov_4D, nbl, npairs):
    """reshape from 4 to 2 dimensions, deprecated. Working but quite convoluted and difficult to read. Not efficient because of
    the various if statements"""
    cov_2D = np.zeros((npairs * nbl, npairs * nbl))
    row = 0
    col = 0
    for ell1 in range(nbl):
        for p in range(npairs):
            col = 0
            if ell1 == 0 and p == 0:
                row = 0
            else:
                row = row + 1
            for ell2 in range(nbl):
                for q in range(npairs):
                    cov_2D[row, col] = cov_4D[ell1, ell2, p, q]
                    col = col + 1
    return cov_2D


@njit
def cov_4D_to_2D_CLOE_old(cov_4D, nbl, p_max, q_max):
    """same as above, but able to accept different zpairs values, producting non-square 2D blocks. Originally used to
    build 3x2pt_2DCLOE covmat"""
    cov_2D = np.zeros((p_max * nbl, q_max * nbl))
    row = 0
    col = 0
    for ell1 in range(nbl):
        for p in range(p_max):
            col = 0
            if ell1 == 0 and p == 0:
                row = 0
            else:
                row = row + 1
            for ell2 in range(nbl):
                for q in range(q_max):
                    cov_2D[row, col] = cov_4D[ell1, ell2, p, q]
                    col = col + 1
    return cov_2D


def cov_4D_to_2D_3x2pt_CLOE_old(cov_4D, nbl, zbins):
    """Builds 3x2pt_2DCLOE using the old cov_4D_to_2D function, discarded.
    """
    npairs, npairs_asimm, npairs_tot = get_zpairs(zbins)

    lim_1 = npairs
    lim_2 = npairs_asimm + npairs
    lim_3 = npairs_tot

    cov_L_L = cov_4D_to_2D_CLOE_old(cov_4D[:, :, :lim_1, :lim_1], nbl, npairs, npairs)
    cov_L_LG = cov_4D_to_2D_CLOE_old(cov_4D[:, :, :lim_1, lim_1:lim_2], nbl, npairs, npairs_asimm)
    cov_L_G = cov_4D_to_2D_CLOE_old(cov_4D[:, :, :lim_1, lim_2:lim_3], nbl, npairs, npairs)

    cov_LG_L = cov_4D_to_2D_CLOE_old(cov_4D[:, :, lim_1:lim_2, :lim_1], nbl, npairs_asimm, npairs)
    cov_LG_LG = cov_4D_to_2D_CLOE_old(cov_4D[:, :, lim_1:lim_2, lim_1:lim_2], nbl, npairs_asimm, npairs_asimm)
    cov_LG_G = cov_4D_to_2D_CLOE_old(cov_4D[:, :, lim_1:lim_2, lim_2:lim_3], nbl, npairs_asimm, npairs)

    cov_G_L = cov_4D_to_2D_CLOE_old(cov_4D[:, :, lim_2:lim_3, :lim_1], nbl, npairs, npairs)
    cov_G_LG = cov_4D_to_2D_CLOE_old(cov_4D[:, :, lim_2:lim_3, lim_1:lim_2], nbl, npairs, npairs_asimm)
    cov_G_G = cov_4D_to_2D_CLOE_old(cov_4D[:, :, lim_2:lim_3, lim_2:lim_3], nbl, npairs, npairs)

    # make long rows and stack together
    row_1 = np.hstack((cov_L_L, cov_L_LG, cov_L_G))
    row_2 = np.hstack((cov_LG_L, cov_LG_LG, cov_LG_G))
    row_3 = np.hstack((cov_G_L, cov_G_LG, cov_G_G))

    array_2D = np.vstack((row_1, row_2, row_3))

    return array_2D


@njit  # XXX this function is new - still to be thouroughly tested (I don't think it's true)
def cov_2D_to_4D_old(cov_2D, nbl, npairs):
    print('this function is deprecated, please use cov_2D_to_4D instead')
    """reshape from 2 to 4 dimensions"""
    # TODO maybe re-check the corresponding new function?
    cov_4D = np.zeros((nbl, nbl, npairs, npairs))
    row = 0
    col = 0
    for ell1 in range(nbl):
        for p in range(npairs):
            col = 0
            if ell1 == 0 and p == 0:
                row = 0
            else:
                row = row + 1
            for ell2 in range(nbl):
                for q in range(npairs):
                    cov_4D[ell1, ell2, p, q] = cov_2D[row, col]
                    col = col + 1
    return cov_4D


def uncertainties_FM_old(FM, nparams=10, fid=None):
    """
    returns relative *percentage!* error
    """
    if fid is None:
        fid = (0.32, 0.05, 1, 1, 0.67, 0.96, 0.816, 0.55, 1, 1)
    else:
        fid = np.where(fid == 0, 1, fid)  # the fiducial for wa is 0, substitute with 1 to avoid division by zero
        fid = np.where(fid == -1, 1, fid)  # the fiducial for wa is -1, substitute with 1 to avoid negative values

    # fidmn = (0.32, 0.05, 1, 1, 0.67, 0.96, 0.816, 0.06, 0.55, 1) # with massive neutrinos
    FM_inv = np.linalg.inv(FM)
    sigma_FM = np.zeros(nparams)
    for i in range(nparams):
        sigma_FM[i] = np.sqrt(FM_inv[i, i]) / fid[i]
    return sigma_FM * 100


def conditional_uncert_FM_old(FM, nparams=10, fid=None):
    """
    returns relative *percentage!* error
    """
    if fid is None:
        fid = (0.32, 0.05, 1, 1, 0.67, 0.96, 0.816, 0.55, 1, 1)
    else:
        fid = np.where(fid == 0, 1, fid)  # the fiducial for wa is 0, substitute with 1 to avoid division by zero
        fid = np.where(fid == -1, 1, fid)  # the fiducial for wa is -1, substitute with 1 to avoid negative values

    # fidmn = (0.32, 0.05, 1, 1, 0.67, 0.96, 0.816, 0.06, 0.55, 1) # with massive neutrinos
    sigma_FM = np.sqrt(1 / np.diag(FM))[:nparams] / fid * 100
    return sigma_FM


def Cl_2D_to_3D_symmetric_old(nbl, zbins, Cl_2D):
    Cl_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        k = 0
        for i in range(zbins):
            for j in range(i, zbins):
                Cl_3D[ell, i, j] = Cl_2D[ell, k]
                k += 1
    return Cl_3D


@njit
def datavector_4D_to_3D_old(D_4D, ind, nParams, nbl, npairs):
    """deprecated in favor of FM_utils.dC_4D_to_3D"""
    D_3D = np.zeros((nbl, npairs, nParams))
    for alf in range(nParams):
        for elle in range(nbl):
            for p in range(npairs):
                D_3D[elle, p, alf] = D_4D[elle, ind[p, 2], ind[p, 3], alf]
    return D_3D
