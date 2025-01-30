import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import csv

import spms_patient_class
import serialization

spms_data_pd = pd.read_excel('SPMS_data_2.xlsx')
sheet_keys = list(spms_data_pd.keys())
print(sheet_keys)

mri_data_begin_position = sheet_keys.index('Datum')
print(mri_data_begin_position)

sheet_keys_patient = sheet_keys[:mri_data_begin_position]
sheet_keys_mri = sheet_keys[mri_data_begin_position:]

patient_list = []
current_patient = spms_patient_class.SPMSPatient('dummy', {})

for i in range(spms_data_pd.shape[0]):
    patient_data_strip = spms_data_pd.iloc[i][sheet_keys[:mri_data_begin_position]].to_dict()
    mri_data_strip = spms_data_pd.iloc[i][sheet_keys[mri_data_begin_position:]].to_dict()
    if patient_data_strip['Patienten-ID'] == patient_data_strip['Patienten-ID']:
        # append old patient, add new patient
        if len(current_patient.mri_image_list) > 1:
            patient_list.append(current_patient)
        current_patient = spms_patient_class.SPMSPatient(patient_id=patient_data_strip['Patienten-ID'],
                                                         patient_data_dict=patient_data_strip)
    #if mri_data_strip['Studien Instance UID'] == mri_data_strip['Studien Instance UID'] and mri_data_strip['Studien Instance UID'] != '-':
    current_patient.add_mri_image(image_id=mri_data_strip['Studien Instance UID'], image_data_dict=mri_data_strip)
patient_list.append(current_patient)
#del patient_list[0]

print('num of patients ', len(patient_list))
print('first patient ', patient_list[0].patient_id)
print('second patient ', patient_list[1].patient_id)
print('last patient ', patient_list[-1].patient_id)
print(len(patient_list[-1].mri_image_list))
print(patient_list[0].mri_image_list[0].image_data_dict["EDSS"])

values_start_end = ["EDSS", "li MEP uE"]  # ["EDSS", "temperature irregulation"]  # , "self-medication"]
values_parsed = sheet_keys[sheet_keys.index(values_start_end[0]):sheet_keys.index(values_start_end[-1]) + 1]
print("values parsed: ", values_parsed)
patient_traj_list, patient_treatment_list = spms_patient_class.parse_raw_values_patient_list(patient_list, values_parsed)

"""for i in range(len(patient_traj_list)):
    print('Patient ID: ', patient_list[i].patient_id)
    print(len(patient_list[i].mri_image_list))
    print(np.shape(patient_traj_list[i]))
"""

patient_traj_list_, patient_list_, data_quality_array = spms_patient_class.normalize_data(patient_traj_list, patient_list, individual_traj=False)
patient_name_list = []
for i in range(len(patient_list)):
    patient_name_list.append(patient_list_[i].patient_id)

print(np.shape(patient_traj_list_[0]))
print(patient_traj_list_[0][:, :5])
print(patient_traj_list_[1][:, :5])
print(patient_traj_list_[2][:, :5])

print(spms_patient_class.compute_frechet_distance(patient_traj_list_[0],
                                                  patient_traj_list_[1],
                                                  oversampling=5))


"""for patient_traj in patient_traj_list_:
    num_data = np.shape(patient_traj)[0]
    ra = np.linspace(start=0, stop=1, num=num_data)
    plt.plot(ra, patient_traj[:, 0], alpha=0.2)
plt.show()"""

big_mat = np.array([]).reshape((-1, np.shape(patient_traj_list_[0])[1]))
traj_ind_in_bm = np.array([])
for patient_ind in range(len(patient_traj_list_)):
    big_mat = np.vstack((big_mat, patient_traj_list_[patient_ind]))
    traj_ind_in_bm = np.hstack((traj_ind_in_bm, np.ones(np.shape(patient_traj_list_[patient_ind])[0]) * patient_ind))

is_traj_member_mat = np.array([]).reshape((-1, np.shape(big_mat)[0]))
for i in range(int(np.max(traj_ind_in_bm)+1)):
    mat_slice = np.zeros(np.shape(big_mat)[0])
    mat_slice[traj_ind_in_bm == i] = 1
    is_traj_member_mat = np.vstack((is_traj_member_mat, mat_slice))
#sign_array = np.ones(np.shape(patient_traj_list_[0])[1])
#sign_array[18] = -1
features_already_chosen = np.zeros(np.shape(patient_traj_list_[0])[1])
print('this should be zero ', np.size(big_mat[big_mat != big_mat]))

print('data loading & preprocessing done')


# transformation into ranking metrics
sign_array = np.ones(np.shape(patient_traj_list_[0])[1])
# sign_array[5] = -1
# sign_array[20] = -1
# m1
"""param_ind_group_list = [[18, 36],
                        [0, 16, 21, 24, 29],
                        [1, 4, 7, 8, 13, 22, 31, 33],
                        [9, 11, 12, 17, 19, 25, 32]]"""
# m2
"""param_ind_group_list = [[1, 18, 23, 36, 37, 38],
                        [0,  3, 15, 16, 21, 24, 29, 34, 35],
                        [7,  9, 10, 11, 14, 19, 26, 27, 31, 33],
                        [4,  6,  8, 12, 13, 17, 20, 22, 25, 32],
                        [5, 28, 30]]"""
# m3
"""param_ind_group_list = [[0, 1, 15, 18, 29],
                        [4, 6, 10, 23, 25, 28, 37],
                        [5, 8, 9, 16, 19],
                        [11, 12, 20, 22, 26, 27, 30, 34],
                        [3, 21, 31, 38]]"""
# m4 +-
param_ind_group_list = [[18, 30, 43],
                        [0, 28, 44],
                        [5, 7, 8, 9, 31, 40],
                        [11, 13, 19, 27, 35, 42, 46],
                        [4, 17, 24, 33, 39, 41, 45]]
sign_array[43] = -1
sign_array[31] = -1
sign_array[11] = -1
sign_array[19] = -1

new_patient_traj_list = spms_patient_class.transform_into_ranking(big_mat, sign_array, traj_ind_in_bm, param_ind_group_list[:2])

print(np.shape(new_patient_traj_list[0]))
print(new_patient_traj_list[0])
print(new_patient_traj_list[1])
print(new_patient_traj_list[2])

print([values_parsed[i] for i in param_ind_group_list[0]])
print([values_parsed[i] for i in param_ind_group_list[1]])
print([values_parsed[i] for i in param_ind_group_list[2]])
print([values_parsed[i] for i in param_ind_group_list[3]])
print([values_parsed[i] for i in param_ind_group_list[4]])

dist_mat = spms_patient_class.construct_traj_dist_mat(new_patient_traj_list, method='frechet', order=2, oversampling=5)
ordered_dist_mat, res_order, res_linkage = serialization.compute_serial_matrix(dist_mat, 'ward')

plt.pcolormesh(ordered_dist_mat)
plt.xlim([0, 120])
plt.ylim([0, 120])
cbar = plt.colorbar()
plt.xlabel('Patient ID')
plt.ylabel('Patient ID')
cbar.set_label('Distance', rotation=90)
plt.show()

# hierarchical clustering
dist_mat_cond = dist_mat[np.triu_indices(np.shape(dist_mat)[0], 1)]
Z = linkage(dist_mat_cond, 'ward')
dendrogram(Z, leaf_rotation=0, leaf_font_size=8)
plt.xticks([])
plt.ylabel('Distance')
plt.xlabel('Patient ID')
plt.show()

print(np.shape(Z))
cluster_ind = fcluster(Z, t=190, criterion='distance')  # 190
print(cluster_ind)


spms_patient_class.plot_all_traj_2d(new_patient_traj_list)

spms_patient_class.plot_all_traj_2d(new_patient_traj_list, cluster_ind=cluster_ind)


# print cluster raw info
writer_list = []
for i in range(np.max(cluster_ind)):
    f_op = open('op_cluster_'+str(i)+'.csv', 'w')
    writer = csv.writer(f_op)
    writer_list.append(writer)

dummy_list = []
for i in range(len(sheet_keys_patient)):
    dummy_list.append(' ')

print('they should be equal ', len(patient_list_), np.shape(cluster_ind))

for i in range(len(patient_list_)):
    for j in range(len(patient_list_[i].mri_image_list)):
        row = []
        if j == 0:
            for key in sheet_keys_patient:
                row.append(patient_list_[i].patient_data_dict[key])
        else:
            row = dummy_list.copy()
        for key in sheet_keys_mri:
            row.append(patient_list_[i].mri_image_list[j].image_data_dict[key])
        #print(len(row))
        writer_list[cluster_ind[i]-1].writerow(row)






