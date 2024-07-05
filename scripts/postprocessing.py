import nibabel as nib
import numpy as np
from torchmetrics.classification import Dice
import torch

def save(dat, fname):
    dat = dat[:, :, None]
    h = nib.Nifti1Header()
    h.set_data_dtype(dat.dtype)
    nib.save(nib.Nifti1Image(dat, np.eye(4), h), fname)

epoch_number = 58990
# best epoch is 9510 with an average dice of 0.8900
experiment = '010_05_bias_1_12'

real_image  = nib.load('../experiments/experiment_' + str(experiment) + '/lightning_logs/version_0/images/epoch-' + str(epoch_number).zfill(5) + '_real-image.nii.gz')
real_pred   = nib.load('../experiments/experiment_' + str(experiment) + '/lightning_logs/version_0/images/epoch-' + str(epoch_number).zfill(5) + '_real-pred.nii.gz')
real_ref    = nib.load('../experiments/experiment_' + str(experiment) + '/lightning_logs/version_0/images/epoch-' + str(epoch_number).zfill(5) + '_real-ref.nii.gz')

# synth0_image = nib.load('../experiment_' + str(experiment) + '/lightning_logs/version_0/images/epoch-' + str(epoch_number).zfill(4) + '_synth0-image.nii.gz')
# synth0_pred  = nib.load('../experiment_' + str(experiment) + '/lightning_logs/version_0/images/epoch-' + str(epoch_number).zfill(4) + '_synth0-pred.nii.gz')

synth_image = nib.load('../experiments/experiment_' + str(experiment) + '/lightning_logs/version_0/images/epoch-' + str(epoch_number).zfill(5) + '_synth-image.nii.gz')
synth_pred  = nib.load('../experiments/experiment_' + str(experiment) + '/lightning_logs/version_0/images/epoch-' + str(epoch_number).zfill(5) + '_synth-pred.nii.gz')
synth_ref   = nib.load('../experiments/experiment_' + str(experiment) + '/lightning_logs/version_0/images/epoch-' + str(epoch_number).zfill(5) + '_synth-ref.nii.gz')

real_image_data = real_image.get_fdata()
real_pred_data  = real_pred.get_fdata()
real_ref_data   = real_ref.get_fdata()

synth_image_data= synth_image.get_fdata()
synth_pred_data = synth_pred.get_fdata()
synth_ref_data  = synth_ref.get_fdata()

avg_dice_real = 0
avg_dice_synth = 0

for i in range(real_image_data.shape[3]):
    # import pdb; pdb.set_trace()
    save(real_image_data[:,:,:,i], '../results/' + 'sample' + str(i) + '_real_image.nii.gz')
    save(real_ref_data[:,:,:,i], '../results/' + 'sample' + str(i) + '_real_label.nii.gz')
    save(np.argmax(real_pred_data[:,:,:,:,i], axis=3).astype(np.float64), '../results/' + 'sample' + str(i) + '_real_pred.nii.gz')

    save(synth_image_data[:,:,:,i], '../results/' + 'sample' + str(i) + '_synth_image.nii.gz')
    save(synth_ref_data[:,:,:,i], '../results/' + 'sample' + str(i) + '_synth_label.nii.gz')
    save(np.argmax(synth_pred_data[:,:,:,:,i], axis=3).astype(np.float64), '../results/' + 'sample' + str(i) + '_synth_pred.nii.gz')

    dice = Dice(average='micro', ignore_index = 0)
    dice_real = dice(torch.tensor(np.argmax(real_pred_data[:,:,0,:,i], axis=2).astype(np.int64)), torch.tensor(real_ref_data[:,:,0,i].astype(np.int64)))
    avg_dice_real += dice_real

    # import pdb; pdb.set_trace()
    dice_synth = dice(torch.tensor(np.argmax(synth_pred_data[:,:,0,:,i], axis=2).astype(np.int64)), torch.tensor(synth_ref_data[:,:,0,i].astype(np.int64)))
    avg_dice_synth += dice_synth
    print('Sample {} has a dice score {}'.format(str(i), str(dice_real)) )

avg_dice_real /= real_image_data.shape[3]
avg_dice_synth /= synth_image_data.shape[3]

print('average dice for real images:', avg_dice_real)
print('average dice for synthetic images:', avg_dice_synth)
# import pdb; pdb.set_trace()