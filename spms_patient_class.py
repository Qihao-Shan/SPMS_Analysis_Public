import numpy as np
import math
import matplotlib.pyplot as plt
import copy


class SPMSPatient:
    def __init__(self, patient_id, patient_data_dict):
        # raw info
        self.patient_id = patient_id
        self.patient_data_dict = patient_data_dict
        self.mri_image_list = []

    def add_mri_image(self, image_id, image_data_dict):
        im = MRIImage(image_id, image_data_dict)
        self.mri_image_list.append(im)


def normalize_data(patient_traj_list, patient_list, individual_traj=False):
    patient_list_ = []
    patient_traj_list_ = []
    big_mat = np.array([]).reshape((-1, np.shape(patient_traj_list[0])[1]))
    for patient_ind in range(len(patient_traj_list)):
        big_mat = np.vstack((big_mat, patient_traj_list[patient_ind]))
    data_quality_array = np.zeros_like(big_mat)
    data_quality_array[big_mat == big_mat] = 1
    data_quality_array = np.sum(data_quality_array, axis=0) / np.shape(big_mat)[0]
    norm_mean = np.nanmean(big_mat, axis=0)
    norm_std = np.nanstd(big_mat, axis=0)
    norm_std_ = np.copy(norm_std)
    norm_std_[norm_std_ == 0] = 1
    for patient_ind in range(len(patient_traj_list)):
        new_traj = (np.copy(patient_traj_list[patient_ind]) - norm_mean) / norm_std
        # fill nan values
        # downward pass
        new_traj_downward = np.copy(new_traj)
        for row_id in range(np.shape(new_traj)[0] - 1):
            row = new_traj_downward[row_id, :]
            row_ = new_traj_downward[row_id + 1, :]
            row_[row_ != row_] = row[row_ != row_]
            new_traj_downward[row_id + 1, :] = row_
        # upward pass
        new_traj_upward = np.copy(new_traj)
        for row_id in range(np.shape(new_traj)[0] - 1):
            row = new_traj_upward[np.shape(new_traj)[0] - 1 - row_id, :]
            row_ = new_traj_upward[np.shape(new_traj)[0] - 1 - row_id - 1, :]
            row_[row_ != row_] = row[row_ != row_]
            new_traj_upward[np.shape(new_traj)[0] - 1 - row_id - 1, :] = row_
        new_traj_upward[new_traj_upward != new_traj_upward] = new_traj_downward[new_traj_upward != new_traj_upward]
        new_traj_downward[new_traj_downward != new_traj_downward] = new_traj_upward[
            new_traj_downward != new_traj_downward]
        new_traj = (new_traj_upward + new_traj_downward) / 2
        # new_traj[new_traj != new_traj] = 0
        # for j in range(np.shape(new_traj)[1]):
        #    traj_clm = (new_traj[:, j] - norm_mean[j]) / norm_std_[j]
        #    traj_clm[traj_clm != traj_clm] = 0
        #    new_traj[:, j] = traj_clm
        new_traj[new_traj != new_traj] = 0
        if np.shape(new_traj)[0] > 1 and new_traj.ndim > 1 and new_traj.size > 0:
            patient_traj_list_.append(new_traj)
            patient_list_.append(patient_list[patient_ind])
    return patient_traj_list_, patient_list_, data_quality_array


"""def normalize_data_only_1st(patient_traj_list):
    patient_traj_list_ = []
    big_mat = np.array([]).reshape((-1, np.shape(patient_traj_list[0])[1]))
    for patient_ind in range(len(patient_traj_list)):
        big_mat = np.vstack((big_mat, patient_traj_list[patient_ind]))
    norm_mean = np.nanmean(big_mat, axis=0)
    norm_std = np.nanstd(big_mat, axis=0)
    for patient_ind in range(len(patient_traj_list)):
        new_traj = np.copy(patient_traj_list[patient_ind])
        new_traj[:, 0] = (new_traj[:, 0]-norm_mean[0])/norm_std[0]
        patient_traj_list_.append(new_traj)
    return patient_traj_list_


def fill_data(patient_traj_list):
    # fill nan values with 0
    patient_traj_list_ = []
    for patient_traj in patient_traj_list:
        new_patient_traj = np.copy(patient_traj)
        new_patient_traj[new_patient_traj != new_patient_traj] = 0
        patient_traj_list_.append(new_patient_traj)
    return patient_traj_list_"""


class MRIImage:
    def __init__(self, image_id, image_data_dict):
        self.image_id = image_id
        self.image_data_dict = image_data_dict


def parse_raw_values_patient_list(patient_list, values_parsed, treatment_data_parsed=None):
    patient_traj_list = []
    patient_treatment_list = []
    for i in range(len(patient_list)):
        traj_mat = np.array([]).reshape((-1, len(values_parsed)))
        if treatment_data_parsed is not None:
            treatment_mat = np.array([]).reshape((-1, len(treatment_data_parsed)))
        else:
            treatment_mat = np.array([])
        for j in range(len(patient_list[i].mri_image_list)):
            traj_slice = extract_slice(patient_list[i].mri_image_list[j], values_parsed)
            if treatment_data_parsed is not None:
                treatment_slice = extract_slice(patient_list[i].mri_image_list[j], treatment_data_parsed)
            else:
                treatment_slice = np.array([])
            if np.any(traj_slice == traj_slice):
                traj_mat = np.vstack((traj_mat, traj_slice))
                treatment_mat = np.vstack((treatment_mat, treatment_slice))
        patient_traj_list.append(traj_mat)
        patient_treatment_list.append(treatment_mat)
    return patient_traj_list, patient_treatment_list


def extract_slice_treatment(mri_image_entry, values_parsed):
    traj_slice = np.zeros(len(values_parsed))
    for k in range(len(values_parsed)):
        value_read = mri_image_entry.image_data_dict[values_parsed[k]]
        if type(value_read) == str:
            ind_ja = value_read.find("ja")
            ind_nein = value_read.find("nein")
            if ind_ja >= 0:
                traj_slice[k] = 1
            elif ind_nein >= 0:
                traj_slice[k] = 0
            elif value_read != value_read or value_read == "-" or value_read.find(
                    '-') >= 0 or value_read == " ":
                traj_slice[k] = 0
            else:
                traj_slice[k] = 1
        else:
            if value_read != value_read:
                traj_slice[k] = 0
            else:
                traj_slice[k] = value_read


def extract_slice(mri_image_entry, values_parsed):
    traj_slice = np.zeros(len(values_parsed))
    for k in range(len(values_parsed)):
        value_read = mri_image_entry.image_data_dict[values_parsed[k]]
        # parse values according to name
        if values_parsed[k] == "EDSS":
            if type(value_read) == int or type(value_read) == float:
                traj_slice[k] = value_read
            elif value_read != value_read or value_read == "-":
                traj_slice[k] = float('nan')
            else:
                if value_read.find('-') < 0:
                    traj_slice[k] = float(value_read)
                else:
                    value_read_list = value_read.split('-')
                    # if len(value_read_list) < 2:
                    #    print(value_read)
                    traj_slice[k] = (float(value_read_list[0]) + float(value_read_list[1])) / 2
        elif values_parsed[k] == "Gehstrecke (m)":
            if value_read != value_read or value_read == "-":
                traj_slice[k] = float('nan')
            else:
                traj_slice[k] = value_read
        elif values_parsed[k] == "re VEP (ms)" or values_parsed[k] == "li VEP (ms)" or values_parsed[k] == "re Tibialis-SEP (ms) P40 (N1)" or values_parsed[k] == "li Tibialis-SEP (ms) P40 (N1)" or values_parsed[k] == "re Medianus-SEP (ms) N20 (N1)" or values_parsed[k] == "li Medianus-SEP (ms) N20 (N1)" or values_parsed[k] == "re MEP oE" or values_parsed[k] == "li MEP oE" or values_parsed[k] == "re MEP uE" or values_parsed[k] == "li MEP uE":
            if value_read != value_read or type(value_read) == str:
                traj_slice[k] = float('nan')
            else:
                traj_slice[k] = value_read
        else:
            # normal parse
            # ja -> 2, nein -> -1
            # no info nan -> 0
            # else -> 1
            if type(value_read) == str:
                ind_ja = value_read.find("ja")
                ind_nein = value_read.find("nein")
                if ind_ja >= 0:
                    traj_slice[k] = 1
                elif ind_nein >= 0:
                    traj_slice[k] = -1
                elif value_read != value_read or value_read == "-" or value_read.find(
                        '-') >= 0 or value_read == " ":
                    traj_slice[k] = float('nan')
                else:
                    traj_slice[k] = 1
            # elif type(value_read) == int or type(value_read) == float:
            #    traj_slice[k] = value_read
    return traj_slice


def parse_patient_list_for_treatment_data(patient_list, values_parsed):
    patient_treatment_mat = np.array([]).reshape((-1, len(values_parsed)))
    for i in range(len(patient_list)):
        traj_mat = np.array([]).reshape((-1, len(values_parsed)))
        for j in range(len(patient_list[i].mri_image_list)):
            traj_slice = np.zeros(len(values_parsed))
            for k in range(len(values_parsed)):
                value_read = patient_list[i].mri_image_list[j].image_data_dict[values_parsed[k]]
                # normal parse
                # ja -> 2, nein -> -1
                # no info nan -> 0
                # else -> 1
                if type(value_read) == str:
                    ind_ja = value_read.find("ja")
                    ind_nein = value_read.find("nein")
                    if ind_ja >= 0:
                        traj_slice[k] = 1
                    elif ind_nein >= 0:
                        traj_slice[k] = -1
                    elif value_read != value_read or value_read == "-" or value_read.find(
                            '-') >= 0 or value_read == " ":
                        traj_slice[k] = float('nan')
                    else:
                        traj_slice[k] = 1
                else:
                    if value_read == value_read:
                        traj_slice[k] = 1
                    else:
                        traj_slice[k] = float('nan')
            traj_mat = np.vstack((traj_mat, traj_slice))
        # fill empty values
        # downward pass
        new_traj_downward = np.copy(traj_mat)
        for row_id in range(np.shape(traj_mat)[0] - 1):
            row = new_traj_downward[row_id, :]
            row_ = new_traj_downward[row_id + 1, :]
            row_[row_ != row_] = row[row_ != row_]
            new_traj_downward[row_id + 1, :] = row_
        # upward pass
        new_traj_upward = np.copy(traj_mat)
        for row_id in range(np.shape(traj_mat)[0] - 1):
            row = new_traj_upward[np.shape(traj_mat)[0] - 1 - row_id, :]
            row_ = new_traj_upward[np.shape(traj_mat)[0] - 1 - row_id - 1, :]
            row_[row_ != row_] = row[row_ != row_]
            new_traj_upward[np.shape(traj_mat)[0] - 1 - row_id - 1, :] = row_
        new_traj_upward[new_traj_upward != new_traj_upward] = new_traj_downward[new_traj_upward != new_traj_upward]
        new_traj_downward[new_traj_downward != new_traj_downward] = new_traj_upward[
            new_traj_downward != new_traj_downward]
        new_traj = (new_traj_upward + new_traj_downward) / 2
        new_traj[new_traj != new_traj] = 0
        patient_treatment_slice = np.nanmean(new_traj, axis=0)
        patient_treatment_mat = np.vstack((patient_treatment_mat, patient_treatment_slice))
    return patient_treatment_mat


def oversample(traj, oversampling):
    m = np.shape(traj)[0]
    if len(np.shape(traj)) > 1:
        n = np.shape(traj)[1]
    else:
        n = 1
    traj_ = traj[0]
    for i in range(m - 1):
        intermediate_step = (traj[i + 1] - traj[i]) / (oversampling + 1)
        traj_additional_samples = np.tile(intermediate_step, (oversampling, 1))
        ra = np.tile((np.arange(oversampling) + 1), (n, 1)).T
        traj_additional_samples = traj_additional_samples * ra
        traj_ = np.vstack((traj_, traj_additional_samples + traj[i]))
        traj_ = np.vstack((traj_, traj[i + 1]))
    return traj_


def _c(ca, i, j, p, q, order=2):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = np.linalg.norm(p[i] - q[j], ord=order)
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i - 1, 0, p, q), np.linalg.norm(p[i] - q[j], ord=order))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j - 1, p, q), np.linalg.norm(p[i] - q[j], ord=order))
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(
                _c(ca, i - 1, j, p, q),
                _c(ca, i - 1, j - 1, p, q),
                _c(ca, i, j - 1, p, q)
            ),
            np.linalg.norm(p[i] - q[j], ord=order)
        )
    else:
        ca[i, j] = float('inf')
    return ca[i, j]


def compute_frechet_distance(traj_0, traj_1, order=2, oversampling=0):
    # remove nan
    # traj_0[traj_0 != traj_0] = 0
    # traj_1[traj_1 != traj_1] = 0
    # oversampling
    # print(traj_1)
    traj_0_ = oversample(traj_0, oversampling)
    traj_1_ = oversample(traj_1, oversampling)
    # normalization
    """traj_0_ = (traj_0_ - np.nanmean(traj_0_)) / np.nanstd(traj_0_)
    traj_0_[traj_0_ != traj_0_] = 0
    traj_1_ = (traj_1_ - np.nanmean(traj_1_)) / np.nanstd(traj_1_)
    traj_1_[traj_1_ != traj_1_] = 0"""
    (m_0, n) = np.shape(traj_0_)
    (m_1, _) = np.shape(traj_1_)
    ca = (np.ones((m_0, m_1), dtype=np.float64) * -1)
    dist = _c(ca, m_0 - 1, m_1 - 1, traj_0_, traj_1_, order=order)
    return dist


def compute_dtw_distance(traj_0, traj_1, order=2, oversampling=0):
    traj_0_ = oversample(traj_0, oversampling)
    traj_1_ = oversample(traj_1, oversampling)
    cost_mat = np.ones((np.shape(traj_0_)[0] + 1, np.shape(traj_1_)[0] + 1)) * float('nan')
    cost_mat[0, 0] = 0
    for i in range(np.shape(traj_0_)[0]):
        for j in range(np.shape(traj_1_)[0]):
            cost = np.linalg.norm(traj_0_[i] - traj_1_[j], ord=order)  # euclidean distance
            cost_mat[i + 1, j + 1] = cost + np.nanmin(
                np.array([cost_mat[i, j], cost_mat[i, j + 1], cost_mat[i + 1, j]]))
    return cost_mat[-1, -1]


def construct_traj_dist_mat(patient_traj_list, method='frechet', order=2, oversampling=0):
    num_pat = len(patient_traj_list)
    fd_mat = np.zeros((num_pat, num_pat))
    for i in range(num_pat):
        for j in range(i):
            if method == 'frechet':
                fd_entry = compute_frechet_distance(patient_traj_list[i],
                                                    patient_traj_list[j],
                                                    order=order,
                                                    oversampling=oversampling)
            else:
                fd_entry = compute_dtw_distance(patient_traj_list[i],
                                                patient_traj_list[j],
                                                order=order,
                                                oversampling=oversampling)
            fd_mat[i, j] = fd_entry
            fd_mat[j, i] = fd_entry
            # print(j)
        print('\rline ', i, '/', num_pat, ' done', end='')
    return fd_mat


def plot_barycentric_traj(patient_traj_list, label_list, plot_end=9999, resort_ind=[]):
    """
    :param patient_traj_list: num_patient * [m * num_metric]
    :param label_list:
    :param plot_end:
    :param resort_ind:
    :return: ref_points on unit circle for vertices, op_coo for centers of mass
    """
    # reorder
    # plot
    big_mat = np.array([]).reshape((-1, np.shape(patient_traj_list[0])[1]))
    ind_array = np.array([])
    for patient_ind in range(len(patient_traj_list)):
        big_mat = np.vstack((big_mat, patient_traj_list[patient_ind]))
        ind_array = np.hstack((ind_array, np.ones(np.shape(patient_traj_list[patient_ind])[0]) * patient_ind))
    if len(resort_ind) > 0:
        big_mat = big_mat[:, resort_ind]
        label_list_ = [label_list[i] for i in resort_ind]
    else:
        label_list_ = label_list
    metric_array_ = np.zeros_like(big_mat) + 0.01
    metric_array_ += (big_mat - np.min(big_mat, axis=0)) / (np.max(big_mat, axis=0) - np.min(big_mat, axis=0))
    metric_array_[metric_array_ != metric_array_] = 0.01
    print(metric_array_)
    ang = 2 * np.pi / np.shape(big_mat)[1]
    ra_ang = np.arange(np.shape(big_mat)[1]) * ang
    x_coo = np.sin(ra_ang)
    y_coo = np.cos(ra_ang)
    ref_points = np.vstack((x_coo, y_coo)).T
    op_coo = metric_array_.dot(ref_points) / np.tile(np.sum(metric_array_, axis=1), (2, 1)).T
    print(op_coo[:5, :])
    # plotting
    fig = plt.figure()
    ax = fig.add_subplot()
    p = ax.scatter(ref_points[:, 0],
                   ref_points[:, 1],
                   c='k', marker='*')
    # q = ax.scatter(op_coo[ind_array < plot_end, 0],
    #               op_coo[ind_array < plot_end, 1],
    #               c=ind_array[ind_array < plot_end], marker='+')
    for i in range(np.shape(ref_points)[0]):
        ax.annotate(label_list_[i], (ref_points[i, 0], ref_points[i, 1]))
    for i in range(int(np.max(ind_array)) + 1):
        if i < plot_end:
            traj_slice = op_coo[ind_array == i, :]
            ax.plot(traj_slice[:, 0], traj_slice[:, 1], alpha=0.2)
            ax.scatter(traj_slice[-1, 0], traj_slice[-1, 1], marker="o")
    plt.show()


def compute_feature_dist_mat(normalized_patient_traj_list):
    normalized_big_mat = np.array([]).reshape((-1, np.shape(normalized_patient_traj_list[0])[1]))
    for patient_ind in range(len(normalized_patient_traj_list)):
        normalized_big_mat = np.vstack((normalized_big_mat, normalized_patient_traj_list[patient_ind]))
    num_features = np.shape(normalized_big_mat)[1]
    cov_mat = np.cov(normalized_big_mat.T)
    std_mat = np.tile(np.std(normalized_big_mat, axis=0), (num_features, 1))
    std_mat_ = std_mat.T
    corr_mat = cov_mat / std_mat / std_mat_
    corr_mat[corr_mat != corr_mat] = 0
    return corr_mat


def extract_criterion(patient_traj_list, label_list, label_summed, sign_array):
    # sum individual metrics into criteria
    ind_list = []
    for label in label_summed:
        ind = label_list.index(label)
        ind_list.append(ind)
    patient_criteria_traj_list = []
    for patient_traj in patient_traj_list:
        # print(ind_list)
        # print(patient_traj[:, ind_list])
        patient_criteria_traj = np.sum(patient_traj[:, ind_list] * sign_array, axis=1)
        # print(patient_criteria_traj)
        patient_criteria_traj_list.append(patient_criteria_traj)
    return patient_criteria_traj_list


def plot_2d_traj(criterion_list_0, criterion_list_1):
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(len(criterion_list_0)):
        plt.plot(criterion_list_0[i], criterion_list_1[i], alpha=0.2)
        plt.scatter(criterion_list_0[i][-1], criterion_list_1[i][-1])
    # ax.set_xlim([-1.7, 2.5])
    # ax.set_ylim([-1, 2])
    plt.show()


def plot_all_traj_2d(patient_traj_list, dim_plotted=(0, 1), cluster_ind=None):
    fig = plt.figure()
    ax = fig.add_subplot()
    color_list = "rbgcmy"
    for i in range(len(patient_traj_list)):
        if cluster_ind is None:
            plt.plot(patient_traj_list[i][:, dim_plotted[0]], patient_traj_list[i][:, dim_plotted[1]])
            plt.scatter(patient_traj_list[i][-1, dim_plotted[0]], patient_traj_list[i][-1, dim_plotted[1]])
        else:
            plt.plot(patient_traj_list[i][:, dim_plotted[0]], patient_traj_list[i][:, dim_plotted[1]], c=color_list[cluster_ind[i]-1], alpha=0.5)
            plt.scatter(patient_traj_list[i][0, dim_plotted[0]], patient_traj_list[i][0, dim_plotted[1]],
                        c=color_list[cluster_ind[i] - 1], marker='x')
            plt.scatter(patient_traj_list[i][-1, dim_plotted[0]], patient_traj_list[i][-1, dim_plotted[1]], c=color_list[cluster_ind[i]-1])
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()


def plot_all_traj_3d(patient_traj_list, dim_plotted=(0, 1, 2), cluster_ind=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    color_list = "rbgcmy"
    for i in range(len(patient_traj_list)):
        if cluster_ind is None:
            ax.plot(patient_traj_list[i][:, dim_plotted[0]], patient_traj_list[i][:, dim_plotted[1]], patient_traj_list[i][:, dim_plotted[2]])
            ax.scatter(patient_traj_list[i][-1, dim_plotted[0]], patient_traj_list[i][-1, dim_plotted[1]], patient_traj_list[i][-1, dim_plotted[2]])
        else:
            ax.plot(patient_traj_list[i][:, dim_plotted[0]], patient_traj_list[i][:, dim_plotted[1]], patient_traj_list[i][:, dim_plotted[2]], c=color_list[cluster_ind[i]-1], alpha=0.5)
            ax.scatter(patient_traj_list[i][0, dim_plotted[0]], patient_traj_list[i][0, dim_plotted[1]], patient_traj_list[i][0, dim_plotted[2]],
                       c=color_list[cluster_ind[i] - 1], marker='x')
            ax.scatter(patient_traj_list[i][-1, dim_plotted[0]], patient_traj_list[i][-1, dim_plotted[1]], patient_traj_list[i][-1, dim_plotted[2]], c=color_list[cluster_ind[i]-1])
    plt.show()


def non_dom_sorting(mat):
    # change to fast sorting
    """
    :param mat: num samples * num features
    :return:
    """
    num_samples = np.shape(mat)[0]
    front_no_array = -np.ones(num_samples)
    current_front = 0
    domination_count = np.zeros(num_samples)
    domination_mat = np.zeros((num_samples, num_samples))
    """while np.min(front_no_array) < 0:
        pareto_front_no_ = front_no_array
        current_front_no = np.max(front_no_array) + 1
        pareto_front_no_[pareto_front_no_ < 0] = current_front_no
        for i in range(np.shape(mat)[0]):
            if pareto_front_no_[i] == current_front_no:
                for j in range(np.shape(mat)[0]):
                    # if i dominate j, remove j
                    if np.all(mat[i] <= mat[j]) and not np.all(
                            mat[i] == mat[j]):
                        pareto_front_no_[j] = -1
        front_no_array = pareto_front_no_"""
    for i in range(np.shape(mat)[0]):
        for j in range(np.shape(mat)[0]):
            if np.all(mat[i] <= mat[j]) and not np.all(mat[i] == mat[j]):
                domination_mat[i, j] = 1
            elif np.all(mat[i] >= mat[j]) and not np.all(mat[i] == mat[j]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            front_no_array[i] = 0
    ra = np.arange(num_samples)
    while np.size(front_no_array[front_no_array == current_front]) != 0:
        for i in ra[front_no_array == current_front]:
            domination_slice = domination_mat[i]
            # print(domination_slice)
            for j in ra[domination_slice == 1]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    front_no_array[j] = current_front + 1
        # print("current front ", current_front)
        # print(front_no_array)
        current_front += 1
    return front_no_array


def parameter_selection_objective_fn(big_mat, feature_sign_array, features_already_chosen):
    """
    :param big_mat:
    :param feature_sign_array:
    :param features_already_chosen: boolean array indicating already chosen features
    :return: objectives array: num features * num_objectives
    objectives: -num fronts, max intra-front spread
    """
    ob_array = np.array([]).reshape((-1, 2))
    ra = np.arange(np.size(features_already_chosen))
    ra = ra[features_already_chosen == 0]
    for i in range(np.size(features_already_chosen[features_already_chosen == 0])):
        features_chosen = np.copy(features_already_chosen)
        features_chosen[ra[i]] = 1
        big_mat_processed = big_mat[:, features_chosen == 1] * feature_sign_array[features_chosen == 1]
        front_no_array = non_dom_sorting(big_mat_processed)
        # print(front_no_array, np.max(front_no_array), np.min(front_no_array))
        neg_num_fronts = -np.max(front_no_array) - 1
        spread_array = -np.ones(int(np.max(front_no_array) + 1))
        pg_global = np.min(big_mat_processed, axis=0)
        pb_global = np.max(big_mat_processed, axis=0)
        denom = np.abs(pb_global - pg_global)
        denom[denom < 0.001] = 0.001
        for j in range(int(np.max(front_no_array) + 1)):
            front_member = big_mat_processed[front_no_array == j]
            pg_front = np.nanmin(front_member, axis=0)
            pb_front = np.nanmax(front_member, axis=0)
            spread_array[j] = np.mean(np.abs(pb_front - pg_front) / denom)
        max_spread = np.max(spread_array)
        if max_spread != max_spread:
            print('issue raised')
            max_spread = 0
        ob_array = np.vstack((ob_array, np.array([neg_num_fronts, max_spread])))
        print("evaluation ", i, " ", [neg_num_fronts, max_spread])
    return ob_array


def parameter_selection_objective_fn_traj_considering(big_mat, is_traj_member_mat, feature_sign_array,
                                                      features_already_chosen):
    """
    :param big_mat
    :param is_traj_member_mat:
    :param feature_sign_array:
    :param features_already_chosen:
    :return:  -mean d front for every trajectory,
    """
    ob_array = np.array([]).reshape((-1, 2))
    ra = np.arange(np.size(features_already_chosen))
    ra = ra[features_already_chosen == 0]
    for i in range(np.size(features_already_chosen[features_already_chosen == 0])):
        features_chosen = np.copy(features_already_chosen)
        features_chosen[ra[i]] = 1
        big_mat_processed = big_mat[:, features_chosen == 1] * feature_sign_array[features_chosen == 1]
        front_no_array = non_dom_sorting(big_mat_processed)
        # neg average max d fronts
        front_no_mat = np.tile(front_no_array, (np.shape(is_traj_member_mat)[0], 1))
        front_no_mat[is_traj_member_mat == 0] = float('nan')
        max_front_no = np.nanmax(front_no_mat, axis=1)
        min_front_no = np.nanmin(front_no_mat, axis=1)
        if np.any(max_front_no - min_front_no < 0):
            print("error===============")
        neg_ave_max_d_fronts = -np.mean(max_front_no - min_front_no)
        ob_array = np.vstack((ob_array, np.array([neg_ave_max_d_fronts, 0])))
        print("evaluation ", i, " ", [neg_ave_max_d_fronts, 0])
    return ob_array


def parameter_selection_objective_fn_2(big_mat, feature_sign_array, features_under_consideration, features_chosen):
    # only neg num fronts
    ob_array = np.array([])
    ra = np.arange(np.size(features_chosen))
    ra = ra[np.logical_and(features_under_consideration == 1, features_chosen == 0)]
    # print('Remaining ', ra)
    for i in range(np.size(ra)):
        features_chosen_ = np.copy(features_chosen)
        features_chosen_[ra[i]] = 1
        big_mat_processed = big_mat[:, features_chosen_ == 1] * feature_sign_array[features_chosen_ == 1]
        front_no_array = non_dom_sorting(big_mat_processed)
        # print(front_no_array, np.max(front_no_array), np.min(front_no_array))
        neg_num_fronts = -np.max(front_no_array) - 1
        ob_array = np.hstack((ob_array, neg_num_fronts))
        #print("evaluation ", ra[i], " ", neg_num_fronts)
    return ob_array


def parameter_selection_objective_fn_3(big_mat, feature_sign_array, features_under_consideration, features_chosen):
    # mean absolute diff between front numbers
    max_front_size_array = np.array([])
    num_front_array = np.array([])
    ra = np.arange(np.size(features_chosen))
    ra = ra[np.logical_and(features_under_consideration == 1, features_chosen == 0)]
    # print('Remaining ', ra)
    for i in range(np.size(ra)):
        features_chosen_ = np.copy(features_chosen)
        features_chosen_[ra[i]] = 1
        big_mat_processed = big_mat[:, features_chosen_ == 1] * feature_sign_array[features_chosen_ == 1]
        front_no_array = non_dom_sorting(big_mat_processed)
        # print(front_no_array, np.max(front_no_array), np.min(front_no_array))
        bin_count = np.bincount(front_no_array.astype(int))
        max_cluster = np.max(bin_count)
        max_front_size_array = np.hstack((max_front_size_array, max_cluster))
        num_front_array = np.hstack((num_front_array, np.max(front_no_array)+1))
        #print("evaluation ", ra[i], " ", max_cluster)
    return max_front_size_array, num_front_array


def parameter_selection_objective_fn_2_traj_considering(big_mat, is_traj_member_mat, feature_sign_array,
                                                        features_under_consideration, features_chosen):
    ob_array = np.array([])
    ra = np.arange(np.size(features_chosen))
    ra = ra[np.logical_and(features_under_consideration == 1, features_chosen == 0)]
    for i in range(np.size(ra)):
        features_chosen_ = np.copy(features_chosen)
        features_chosen_[ra[i]] = 1
        big_mat_processed = big_mat[:, features_chosen_ == 1] * feature_sign_array[features_chosen_ == 1]
        front_no_array = non_dom_sorting(big_mat_processed)
        front_no_mat = np.tile(front_no_array, (np.shape(is_traj_member_mat)[0], 1))
        front_no_mat[is_traj_member_mat == 0] = float('nan')
        max_front_no = np.nanmax(front_no_mat, axis=1)
        min_front_no = np.nanmin(front_no_mat, axis=1)
        neg_ave_max_d_fronts = -np.mean(max_front_no - min_front_no)
        ob_array = np.hstack((ob_array, neg_ave_max_d_fronts))
    return ob_array


def transform_into_ranking(big_mat, features_sign_array, traj_ind_in_bm, param_ind_grp_list):
    # construct new patient_traj, dim: num_patients * [traj_length * num_param_grp]
    # build ranking big mat
    new_big_mat = np.array([]).reshape((-1, np.shape(big_mat)[0]))
    for i in range(len(param_ind_grp_list)):
        big_mat_slice = big_mat[:, param_ind_grp_list[i]] * features_sign_array[param_ind_grp_list[i]]
        ranking_slice = non_dom_sorting(big_mat_slice)
        new_big_mat = np.vstack((new_big_mat, ranking_slice))
    new_big_mat = new_big_mat.T
    # normalize
    # new_big_mat = (new_big_mat - np.mean(new_big_mat, axis=0)) / np.std(new_big_mat, axis=0)
    print(np.shape(new_big_mat))
    # split into traj list
    new_patient_traj_list = []
    for patient_ind in range(int(np.max(traj_ind_in_bm)+1)):
        patient_traj = new_big_mat[traj_ind_in_bm == patient_ind, :]
        new_patient_traj_list.append(patient_traj)
    return new_patient_traj_list


def contribution_scatter_pc12(contribution_array_pc1, contribution_array_pc2, ax, marker, color):
    ax.scatter(contribution_array_pc1, contribution_array_pc2, marker=marker, c=color)
    for i in range(np.size(contribution_array_pc1)):
        ax.annotate(str(i), (contribution_array_pc1[i], contribution_array_pc2[i]))






