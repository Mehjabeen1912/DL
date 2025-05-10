import pandas as pd
import os
import numpy as np
import nibabel as nib

labels_df = pd.read_csv('/home/tasni001/name_mapping.csv')
df = labels_df[['Grade', 'BraTS_2020_subject_ID']]

rootdir = '/home/tasni001/DL_Dataset'
subdirs = os.listdir(rootdir)
label = np.zeros(369)
inputs = np.zeros((369, 160, 210, 100, 1))
label_2d = np.zeros(36900)
inputs_2d = np.zeros((36900, 160, 210, 1))
i = 0
j = 0
for sub in subdirs:
    file_path = rootdir + '/' + sub + '/' + sub + '_weighted1233.nii.gz'
    if os.path.exists(file_path):
        nii_img = nib.load(file_path)
        wt = nii_img.get_fdata()
        intermd = wt[40:200, 20:230, 27:127]
        inputs[i, :, :, :, 0] = intermd

        grade = df.loc[df['BraTS_2020_subject_ID'] == sub, 'Grade'].values[0]
        if grade == 'HGG':
            label[i] = 1
        else:
            label[i] = 0

        for t in range(100):
            inputs_2d[j, :, :, 0] = intermd[:, :, t]
            label_2d[j] = label[i]
            j += 1
        i += 1
#(369, 160, 210, 100, 1)
#369
print(inputs.shape)
print(label.shape)

print(inputs_2d.shape)
print(label_2d.shape)


np.save('/home/tasni001/inputs.npy', inputs)
np.save('/home/tasni001/label.npy', label)

np.save('/home/tasni001/inputs_2d.npy', inputs_2d)
np.save('/home/tasni001/label_2d.npy', label_2d)
