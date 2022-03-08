from shutil import rmtree
import xnat
import os
from os.path import join
import zipfile
from time import sleep
from tqdm import tqdm
import numpy as np

bp = 'D:\\PhD\\Data\\TCGA_new_TB'


# for root, folders, files in os.walk(bp):
    
#     if len(files) > 0:
        
#         dcm_rt_files = [file for file in files if file.startswith('TCGA')]
        
#         if len(dcm_rt_files) > 0:
#             dcm_rt_file = dcm_rt_files[0]
#             os.remove(os.path.join(root, dcm_rt_file))
#             print('removed ', dcm_rt_file)


# %%
VTT_path = 'D:\\PhD\\Data\\VTT'

n_dcmrt_files = 0

for task  in os.listdir(VTT_path):
    
    for scan in os.listdir(os.path.join(VTT_path, task)):
        
        dcm_rt_files = [dcm for dcm in os.listdir(os.path.join(VTT_path, task, scan))
                                                  if dcm.startswith('TCGA')]
        n_dcmrt_files += len(dcm_rt_files)
        dcm_rt_file = dcm_rt_files[0]
        if dcm_rt_file.endswith('TB.dcm'):
            print('removing ', task, scan)
            rmtree(os.path.join(VTT_path, task, scan))
# %%
def upload_roi_dcm(dcmp, session, project_id, subject_id, experiment_id, rt_name_extension):
    
    roi_dcms = get_roi_dcms(dcmp)
    if len(roi_dcms) == 0:
        return
    else:
        print('Uploading roi dcm (%d found).'%len(roi_dcms))
        file_path = roi_dcms[-1]
    
    xnat_project = session.projects[project_id]
    xnat_subject = session.classes.SubjectData(parent=xnat_project,
                                               label=subject_id)
    xnat_experiment = xnat_subject.experiments[experiment_id]
    segmentation_name = subject_id + '_' + rt_name_extension
    segmentation_type = 'RTSTRUCT'
    target_url = (f"/xapi/roi/projects/{project_id}/sessions/{xnat_experiment.id}/collections/{segmentation_name}"f"?type={segmentation_type}&overwrite=true")

    with open(file_path, 'rb') as file:
        response = session.put(target_url, data=file, accepted_status=[200])
        print(response)

xnathost   = 'https://vtt.medschl.cam.ac.uk/'
project_id = 'VisualTuringTest'

user_id    = 'vtt_owner1'
pwd        = 'Passw0rdVTT'#getpass.getpass("Password for user name : %s = " % user_id)

VTT_path = 'D:\\PhD\\Data\\VTT'


session = xnat.connect(xnathost, user=user_id, password=pwd)

# %%

with xnat.connect(xnathost, user=user_id, password=pwd) as session:
    
    dcm_rt_labels = []
    
    for experiment in session.experiments:
        
        xnat_experiment = session.experiments[experiment]
        
        if hasattr(xnat_experiment, 'name'):
            if xnat_experiment.name == 'RTstruct':
                
                label = xnat_experiment.data['label']
                scan = label[:12]
                task = label[13:]
                
                if scan not in os.listdir(os.path.join(VTT_path, task)):
                    dcm_rt_labels.append(xnat_experiment.data['label'])
                    xnat_experiment.delete()


# %%
# with xnat.connect(xnathost, user=user_id, password=pwd) as session:
    
#     CT_sessions = []
    
#     for experiment in session.experiments:
        
#         xnat_experiment = session.experiments[experiment]
        
#         if hasattr(xnat_experiment, 'data'):
#             if 'modality' in xnat_experiment.data:
                
#                 if xnat_experiment.data['modality'] == 'CT':
                
#                     CT_sessions.append(xnat_experiment.data['label'])
                
#     for task in os.listdir(VTT_path):
#         print(task)
#         sleep(0.1)
#         for scan in tqdm(os.listdir(join(VTT_path, task))):
#             dcmp = join(VTT_path, task, scan)
#             subject_id = scan
#             experiment_id = scan + '_CT_' + task
            
#             if experiment_id not in session.experiments:
#                 try:
#                     upload_dcm_folder(dcmp, session, project_id, subject_id, experiment_id)
#                 except:
#                     scan_fails.append(dcmp)
#                     continue
                
#                 try:
#                     upload_roi_dcm(dcmp, session, project_id, subject_id, experiment_id, task)
#                 except:
#                     dcmrt_fails.append(dcmp)
#             else:
#                 print('Skipping ', experiment_id, ' already uploaded')


# if len(scan_fails) > 0:
#     print('The upload failed for the following scans:')
#     print(scan_fails)
# if len(dcmrt_fails) > 0:
#     print('The upload of the dcm_rt file failed for:')
#     print(dcmrt_fails)

# %%
# all_scans = []
# for task in os.listdir(VTT_path):
#     for scan in tqdm(os.listdir(join(VTT_path, task))):
#         all_scans.append(scan)

# un_scans, counts = np.unique(all_scans, return_counts=True)
# for scan, c in zip(un_scans, counts):
#     print(scan + ': '+str(c))