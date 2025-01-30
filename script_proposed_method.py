import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
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

values_start_end = ["EDSS", "Temperatur- irregulation", "li MEP uE"]  # ["EDSS", "temperature irregulation"]  # , "self-medication"]
values_parsed = sheet_keys[sheet_keys.index(values_start_end[0]):sheet_keys.index(values_start_end[-1]) + 1]
print("values parsed: ", values_parsed)
patient_traj_list, patient_treatment_list = spms_patient_class.parse_raw_values_patient_list(patient_list, values_parsed)

"""for i in range(len(patient_traj_list)):
    print('Patient ID: ', patient_list[i].patient_id)
    print(len(patient_list[i].mri_image_list))
    print(np.shape(patient_traj_list[i]))
"""
patient_name_list = []
for patient in patient_list:
    patient_name_list.append(patient.patient_id)

patient_traj_list_, patient_name_list, data_quality_array = spms_patient_class.normalize_data(patient_traj_list, patient_name_list, individual_traj=False)

print(np.shape(patient_traj_list_[0]))
print(patient_traj_list[0][:, :6])
print(patient_traj_list_[0][:, :6])
print(patient_traj_list[1][:, :6])
print(patient_traj_list_[1][:, :6])

print(spms_patient_class.compute_frechet_distance(patient_traj_list_[0],
                                                  patient_traj_list_[1],
                                                  oversampling=5))


big_mat = np.array([]).reshape((-1, np.shape(patient_traj_list_[0])[1]))
traj_ind_in_bm = np.array([])
for patient_ind in range(len(patient_traj_list_)):
    big_mat = np.vstack((big_mat, patient_traj_list_[patient_ind]))
    traj_ind_in_bm = np.hstack((traj_ind_in_bm, np.ones(np.shape(patient_traj_list_[patient_ind])[0]) * patient_ind))
print('big mat shape ', np.shape(big_mat))

is_traj_member_mat = np.array([]).reshape((-1, np.shape(big_mat)[0]))
for i in range(int(np.max(traj_ind_in_bm)+1)):
    mat_slice = np.zeros(np.shape(big_mat)[0])
    mat_slice[traj_ind_in_bm == i] = 1
    is_traj_member_mat = np.vstack((is_traj_member_mat, mat_slice))
# sign_array = np.ones(np.shape(patient_traj_list_[0])[1])
# sign_array[18] = -1
features_already_chosen = np.zeros(np.shape(patient_traj_list_[0])[1])
print('this should be zero ', np.size(big_mat[big_mat != big_mat]))

print('data loading & preprocessing done')


# proposed methods
# trajectory independent dimensionality reduction
# dim reduction from big mat -> clustering front number traj
param_sets_selected = []
sign_array = np.ones(np.shape(patient_traj_list_[0])[1])
#sign_array[18] = -1
ra = np.arange(np.size(sign_array))
features_under_consideration = np.ones(np.shape(patient_traj_list_[0])[1])
i = 0
while np.sum(features_under_consideration) > 1:
    print("Remaining ", ra[features_under_consideration == 1])
    features_already_chosen = np.zeros(np.shape(patient_traj_list_[0])[1])
    num_fronts_old = float('nan')
    stop_cond = False
    while not stop_cond:
        ob_array_pos, num_fronts_array_pos = spms_patient_class.parameter_selection_objective_fn_3(big_mat=big_mat,
                                                                                                   feature_sign_array=sign_array,
                                                                                                   features_under_consideration=features_under_consideration,
                                                                                                   features_chosen=features_already_chosen)
        sign_array_neg = np.copy(sign_array)
        sign_array_neg[features_under_consideration == 1] = -1
        ob_array_neg, num_fronts_array_neg = spms_patient_class.parameter_selection_objective_fn_3(big_mat=big_mat,
                                                                                                   feature_sign_array=sign_array_neg,
                                                                                                   features_under_consideration=features_under_consideration,
                                                                                                   features_chosen=features_already_chosen)
        ob_array_full = np.vstack((ob_array_pos, ob_array_neg))
        num_fronts_array_full = np.vstack((num_fronts_array_pos, num_fronts_array_neg))
        min_front_size = np.min(ob_array_full)
        print(ob_array_full)
        # print(num_fronts_array_full)
        ra_under_consideration_n_not_chosen = ra[np.logical_and(features_under_consideration == 1, features_already_chosen == 0)]
        if num_fronts_old == np.nanmin(np.array([num_fronts_old, min_front_size])):
            stop_cond = True
        else:
            num_fronts_old = min_front_size
            chosen_metric_this_round = ra_under_consideration_n_not_chosen[np.argmin(np.min(ob_array_full, axis=0))]
            sign = np.argmin(np.min(ob_array_full, axis=1))
            if sign == 1:
                sign_array[chosen_metric_this_round] = -1
            print('Chosen Param ', chosen_metric_this_round, sign_array[chosen_metric_this_round], num_fronts_old, num_fronts_array_full[sign, np.argmin(np.min(ob_array_full, axis=0))])
            features_already_chosen[chosen_metric_this_round] = 1
    features_under_consideration[features_already_chosen == 1] = 0
    print("Round ", i, " params chosen: ", ra[features_already_chosen == 1], sign_array[features_already_chosen == 1], num_fronts_old)
    param_sets_selected.append(features_already_chosen)
    i += 1






