import nibabel as nib
import numpy as np
from torchmetrics.classification import Dice
import torch

epoch_number = 1369
experiment = '005_1_1'

best_dice   = 0
best_epoch  = 0
for k in range(200,epoch_number):
    avg_dice    = 0
    real_image  = nib.load('../experiment_' + str(experiment) + '/lightning_logs/version_3805964/images/epoch-' + str(k * 10).zfill(4) + '_real-image.nii.gz')
    real_pred   = nib.load('../experiment_' + str(experiment) + '/lightning_logs/version_3805964/images/epoch-' + str(k * 10).zfill(4) + '_real-pred.nii.gz')
    real_ref    = nib.load('../experiment_' + str(experiment) + '/lightning_logs/version_3805964/images/epoch-' + str(k * 10).zfill(4) + '_real-ref.nii.gz')

    real_image_data = real_image.get_fdata()
    real_pred_data  = real_pred.get_fdata()
    real_ref_data   = real_ref.get_fdata()

    for i in range(real_image_data.shape[3]):
        # import pdb; pdb.set_trace()
        dice = Dice(average='micro', ignore_index = 0)
        score = dice(torch.tensor(np.argmax(real_pred_data[:,:,0,:,i], axis=2).astype(np.int64)), torch.tensor(real_ref_data[:,:,0,i].astype(np.int64)))
        avg_dice += score
        print('In Epoch {} Sample {} has a dice score {}'.format(str(k * 10), str(i), str(score)))

    avg_dice /= real_image_data.shape[3]
    if (avg_dice > best_dice):
        best_dice = avg_dice
        best_epoch = k

print('In Epoch {} has a best average dice score {}'.format(str(best_epoch * 10), str(best_dice)))
import pdb; pdb.set_trace()