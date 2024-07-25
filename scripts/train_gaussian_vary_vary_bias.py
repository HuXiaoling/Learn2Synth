import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
sys.path.append('..')
from learn2synth.networks import UNet, SegNet
from learn2synth.train import LearnableSynthSeg, SynthSeg
from learn2synth.losses import DiceLoss, LogitMSELoss, CatLoss, CatMSELoss
from learn2synth.metrics import Dice, Hausdorff
from learn2synth import optim
from cornucopia import (
    SynthFromLabelTransform, LoadTransform, NonFinalTransform, FinalTransform
)
import cornucopia as cc
from learn2synth.utils import folder2files
from torch.utils.data import Dataset, DataLoader
from typing import Sequence, List, Tuple, Optional, Union
from glob import glob
from os import path, makedirs
from ast import literal_eval
from random import shuffle
import nibabel as nib
import numpy as np
import torch
import math
import fnmatch
import random
import torch.nn.functional as F
from torchmetrics.classification import Dice as dice_compute

class Noisify_Bias_Field(torch.nn.Module):
    """
    An extremely simple synth+ network that just 
    adds scaled Gaussian noise
    """
    def __init__(self):
        super().__init__()
        self.sigma_min = torch.nn.Parameter(torch.rand([]), requires_grad=True)
        self.sigma_max = torch.nn.Parameter(torch.rand([]), requires_grad=True)

        self.weight_low = torch.nn.Parameter(torch.rand([]), requires_grad=True)
        self.weight_middle = torch.nn.Parameter(torch.rand([]), requires_grad=True)
        self.weight_high = torch.nn.Parameter(torch.rand([]), requires_grad=True)
        self.low_bound = 0.5
        self.up_bound = 2

    def forward(self, x):
        bias_field_ori_low = torch.rand([2, 2]).to(x) * (self.up_bound - self.low_bound) + self.low_bound
        bias_field_ori_middle = torch.rand([4, 4]).to(x) * (self.up_bound - self.low_bound) + self.low_bound
        bias_field_ori_high = torch.rand([8, 8]).to(x) * (self.up_bound - self.low_bound) + self.low_bound
        bias_field_low = F.interpolate(bias_field_ori_low.unsqueeze(0).unsqueeze(0), \
                                       size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        bias_field_middle = F.interpolate(bias_field_ori_middle.unsqueeze(0).unsqueeze(0), \
                                          size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        bias_field_high = F.interpolate(bias_field_ori_high.unsqueeze(0).unsqueeze(0), \
                                        size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        low_eps = torch.sigmoid(self.weight_low)
        middle_eps = torch.sigmoid(self.weight_middle)
        high_eps = torch.sigmoid(self.weight_high)
        # low_eps = sigmoid(self.weight_low))
        # set low_eps to fix value in real branch: (0,1): 
        # middle_eps = high_eps = 0; low_eps = 1
        # bias_field_low ** low_eps

        self.sigma = torch.rand([]).to(x) * (self.sigma_max - self.sigma_min) + self.sigma_min
    
        return x * (bias_field_low.squeeze(0) ** low_eps) * \
                (bias_field_middle.squeeze(0) ** middle_eps) * \
                (bias_field_high.squeeze(0) ** high_eps) + torch.randn_like(x) * self.sigma.to(x)


class Model(pl.LightningModule):

    def __init__(self,
                 ndim: int = 2,
                 nb_classes: int = 24,
                 seg_nb_levels: int = 5,
                 seg_features: Sequence[int] = (24, 48, 96, 192, 384, 768),
                 seg_activation: str = 'ELU',
                 seg_nb_conv: int = 2,
                 seg_norm: Optional[str] = None,
                #  synth_nb_levels: int = 1,
                #  synth_features: Tuple[int] = (16,),
                #  synth_activation: str = 'ELU',
                #  synth_nb_conv: int = 1,
                #  synth_norm: Optional[str] = None,
                 synth_residual: bool = True,
                 #  synth_shared: bool = True,
                 loss: str = 'dice',
                 alpha: float = 1.,
                 real_sigma_min: float = 0.15,
                 real_sigma_max: float = 0.15,
                 real_low: float = 0.5, 
                 real_middle: float = 0.5,
                 real_high: float = 0.5,
                 classic: bool = False,
                 optimizer: str = 'Adam',
                 optimizer_options: dict = dict(lr=1e-3),
                 # metrics: dict = dict(dice='dice'),
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer = optimizer
        self.optimizer_options = dict(optimizer_options or {})
        self.alpha = alpha
        self.real_sigma_min = real_sigma_min
        self.real_sigma_max = real_sigma_max
        self.real_low = real_low
        self.real_middle = real_middle
        self.real_high = real_high

        segnet = UNet(
            ndim,
            features=seg_features,
            activation=seg_activation,
            nb_levels=seg_nb_levels,
            nb_conv=seg_nb_conv,
            norm=seg_norm,
        )
        segnet = SegNet(ndim, 1, nb_classes, backbone=segnet, activation=None)

        # synth = SharedSynth if synth_shared else DiffSynth
        # synth = cc.batch(synth(SynthFromLabelTransform(order=1)))
        synth = SynthFromLabelTransform(order=1, resolution=False, snr=False, bias=False)
        synth = cc.batch(DiffSynthFull(synth, real_sigma_min=real_sigma_min, real_sigma_max=real_sigma_max, \
                                       real_low=real_low, real_middle=real_middle, real_high=real_high))

        if loss == 'dice':
            loss = DiceLoss(activation='Softmax')
        elif loss == 'logitmse':
            loss = LogitMSELoss()
        elif loss == 'cat':
            loss = CatLoss(activation='Softmax')
        elif loss == 'catmse':
            loss = CatMSELoss(activation='Softmax')
        elif isinstance(loss, str):
            raise ValueError('Unsupported loss', loss)

        # metrics = metrics or {}
        # for key, val in metrics:
        #     if val == 'dice':
        #         val = Dice()
        #     elif val == 'hausdorff95':
        #         val = Hausdorff(pct=0.95)
        #     elif val == 'hausdorff':
        #         val = Hausdorff()
        #     elif isinstance(val, str):
        #         raise ValueError('Unsupported loss', loss)
        #     metrics[key] = val
        # self.metrics = metrics

        self.classic = classic
        if self.classic:
            self.network = SynthSeg(segnet, synth, loss)
        else:
            synthnet = Noisify_Bias_Field()
            # synthnet = UNet(
            #     ndim,
            #     features=synth_features,
            #     activation=synth_activation,
            #     nb_levels=synth_nb_levels,
            #     nb_conv=synth_nb_conv,
            #     norm=synth_norm,
            # )
            # synthnet = SegNet(ndim, 1, 1, backbone=synthnet, activation=None)

            self.network = LearnableSynthSeg(segnet, synth, synthnet, loss, alpha,
                                             residual=synth_residual)

        self.automatic_optimization = False
        self.network.set_backward(self.manual_backward)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optimizer)
        optimizer_init = lambda x: optimizer(x, **(self.optimizer_options or {}))
        optimizers = self.network.configure_optimizers(optimizer_init)
        self.network.set_optimizers(self.optimizers)
        return optimizers

    def training_step(self, batch, batch_idx):
        # self.trainer.fit_loop.max_epochs = 10000
        if self.trainer.current_epoch % 10 == 0 and batch_idx == 0:
            torch.cuda.empty_cache()

        loss_synth, loss_real = self.network.synth_and_train_step(*batch)
        loss = loss_synth + self.alpha * loss_real
        name = type(self.network.loss).__name__
        # self.log(f'train_loss_synth_{name}', loss_synth)
        # self.log(f'train_loss_real_{name}', loss_real)
        # self.log(f'train_loss_{name}', loss)
        self.log(f'train_loss', loss)
        self.log(f'sigma_min',self.network.synthnet.sigma_min, prog_bar=True)
        self.log(f'sigma_max',self.network.synthnet.sigma_max, prog_bar=True)
        self.log(f'low_coefficient', torch.sigmoid(self.network.synthnet.weight_low), prog_bar=True)
        self.log(f'middle_coefficient', torch.sigmoid(self.network.synthnet.weight_middle), prog_bar=True)
        self.log(f'high_coefficient', torch.sigmoid(self.network.synthnet.weight_high), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        name = type(self.network.loss).__name__
        dice_real = 0
        if self.classic:
            if batch_idx == 0:
                root = f'{self.logger.log_dir}/images'
                makedirs(root, exist_ok=True)
                epoch = self.trainer.current_epoch
                loss_synth, loss_real, pred_synth, pred_real, \
                synth_image, synth_ref, real_image, real_ref \
                    = self.network.synth_and_eval_for_plot(*batch)
                if epoch % 10 == 0:
                    # save(pred_synth.softmax(1).movedim(0, -1).movedim(0, -2),
                    #     f'{root}/epoch-{epoch:04d}_synth-pred.nii.gz')
                    # save(pred_real.softmax(1).movedim(0, -1).movedim(0, -2),
                    #     f'{root}/epoch-{epoch:04d}_real-pred.nii.gz')
                    # save(synth_image.squeeze(1).movedim(0, -1),
                    #     f'{root}/epoch-{epoch:04d}_synth-image.nii.gz')
                    # save(real_image.squeeze(1).movedim(0, -1),
                    #     f'{root}/epoch-{epoch:04d}_real-image.nii.gz')
                    # save(synth_ref.squeeze(1).movedim(0, -1).to(torch.uint8),
                    #     f'{root}/epoch-{epoch:04d}_synth-ref.nii.gz')
                    # save(real_ref.squeeze(1).movedim(0, -1).to(torch.uint8),
                    #     f'{root}/epoch-{epoch:04d}_real-ref.nii.gz')
                    
                    for i in range(real_ref.squeeze(1).movedim(0, -1).shape[2]):
                        dice_score = dice_compute(average='micro', ignore_index = 0)
                        dice_real += dice_score(np.argmax(pred_real[i,].cpu(), axis=0), real_ref[i,:,:].cpu())
                    
                    dice_real /= real_ref.squeeze(1).movedim(0, -1).shape[2]
                    self.log(f'dice_real', dice_real, prog_bar=True)
            else:
                loss_synth, loss_real = self.network.synth_and_eval_step(*batch)
        else:
            if batch_idx == 0:
                root = f'{self.logger.log_dir}/images'
                makedirs(root, exist_ok=True)
                epoch = self.trainer.current_epoch
                loss_synth, loss_synth0, loss_real, \
                pred_synth, pred_synth0, pred_real, \
                synth_image, synth0_image, synth_ref, real_image, real_ref \
                    = self.network.synth_and_eval_for_plot(*batch)
                if epoch % 10 == 0:
                    # save(pred_synth.softmax(1).movedim(0, -1).movedim(0, -2),
                    #     f'{root}/epoch-{epoch:04d}_synth-pred.nii.gz')
                    # save(pred_synth0.softmax(1).movedim(0, -1).movedim(0, -2),
                    #     f'{root}/epoch-{epoch:04d}_synth0-pred.nii.gz')
                    # save(pred_real.softmax(1).movedim(0, -1).movedim(0, -2),
                    #     f'{root}/epoch-{epoch:04d}_real-pred.nii.gz')
                    # save(synth_image.squeeze(1).movedim(0, -1),
                    #     f'{root}/epoch-{epoch:04d}_synth-image.nii.gz')
                    # save(synth0_image.squeeze(1).movedim(0, -1),
                    #     f'{root}/epoch-{epoch:04d}_synth0-image.nii.gz')
                    # save(real_image.squeeze(1).movedim(0, -1),
                    #     f'{root}/epoch-{epoch:04d}_real-image.nii.gz')
                    # save(synth_ref.squeeze(1).movedim(0, -1).to(torch.uint8),
                    #     f'{root}/epoch-{epoch:04d}_synth-ref.nii.gz')
                    # save(real_ref.squeeze(1).movedim(0, -1).to(torch.uint8),
                    #     f'{root}/epoch-{epoch:04d}_real-ref.nii.gz')
                    
                    for i in range(real_ref.squeeze(1).movedim(0, -1).shape[2]):
                        dice_score = dice_compute(average='micro', ignore_index = 0)
                        dice_real += dice_score(np.argmax(pred_real[i,].cpu(), axis=0), real_ref[i,:,:].cpu())
                    
                    dice_real /= real_ref.squeeze(1).movedim(0, -1).shape[2]
                    self.log(f'dice_real', dice_real, prog_bar=True)
            else:
                loss_synth, loss_synth0, loss_real = self.network.synth_and_eval_step(*batch)
            # self.log(f'eval_loss_synth0_{name}', loss_synth0)
        loss = loss_synth + self.alpha * loss_real
        # self.log(f'eval_loss_synth_{name}', loss_synth)
        # self.log(f'eval_loss_real_{name}', loss_real)
        # self.log(f'eval_loss_{name}', loss)
        self.log(f'eval_loss', loss)
        return loss

    def forward(self, x):
        return self.network(x)


def save(dat, fname):
    dat = dat[:, :, None]
    dat = dat.detach().cpu().numpy()
    h = nib.Nifti1Header()
    h.set_data_dtype(dat.dtype)
    nib.save(nib.Nifti1Image(dat, np.eye(4), h), fname)


class SharedSynth(torch.nn.Module):
    """Apply the same geometric transform for synth and real"""

    def __init__(self, synth):
        super().__init__()
        self.synth = synth

    def forward(self, slab, img, lab):
        final = self.synth.make_final(slab, 1)
        final.deform = final.deform.make_final(slab)
        simg, slab = final(slab)
        rimg, rlab = final.deform([img, lab])
        rimg = final.intensity(rimg)
        rlab = final.postproc(rlab)
        return simg, slab, rimg, rlab


class DiffSynth(torch.nn.Module):
    """Apply different geometric transform for synth and real"""

    def __init__(self, synth):
        super().__init__()
        self.synth = synth

    def forward(self, slab, img, lab):
        # slab: labels of the source (synth) domain
        # img: image of the target (real) domain
        # lab: label of the target (real) domain
        final = self.synth.make_final(slab, 1)
        final.deform = final.deform.make_final(slab)
        simg, slab = final(slab)
        rimg, rlab = final.deform(img, lab)
        rimg = final.intensity(img)
        rlab = final.postproc(rlab)
        return simg, slab, rimg, rlab


class DiffSynthFull(torch.nn.Module):
    """
    Generate two fully synthetic images.
    One of them (the target) has noise.
    The other (the source) does not have noise.
    """

    def __init__(self, synth, real_sigma_min=0.15, real_sigma_max=0.15, real_low=0.5, real_middle=0.5, real_high=0.5):
        super().__init__()
        self.synth = synth
        self.real_sigma_min = real_sigma_min
        self.real_sigma_max = real_sigma_max
        self.real_low = real_low
        self.real_middle = real_middle
        self.real_high = real_high

    def forward(self, slab, _, tlab):
        # slab: labels of the source (synth) domain
        # tlab: label of the target (real) domain

        # synthetic real images = noise_free_image * bias filed + noise
        # Bias field = ((2*2 -> upsampling to 256*256) ** (eps_a * a) * ((4*4 -> upsampling to 256*256) 
        # ** (eps_b * b)) * ((8*8 -> upsampling to 256*256) ** (eps_c * c))
        
        real_sigma = random.uniform(self.real_sigma_min, self.real_sigma_max)
        low_bound = 0.5
        up_bound = 2

        bias_field_ori_low = torch.rand([2, 2]) * (up_bound - low_bound) + low_bound
        bias_field_ori_middle = torch.rand([4, 4]) * (up_bound - low_bound) + low_bound
        bias_field_ori_high = torch.rand([8, 8]) * (up_bound - low_bound) + low_bound

        # low_eps = sigmoid(self.weight_low))
        # set low_eps to fix value in real branch: (0,1): 
        # middle_eps = high_eps = 0; low_eps = 1
        # bias_field_low ** low_eps
        # return x * (bias_field_low.squeeze(0) ** low_eps) * \
        #         (bias_field_middle.squeeze(0) ** middle_eps) * \
        #         (bias_field_high.squeeze(0) ** high_eps) + torch.randn_like(x) * self.sigma.to(x)
    
        simg, slab = self.synth(slab)
        timg, tlab = self.synth(tlab)

        bias_field_low = F.interpolate(bias_field_ori_low.unsqueeze(0).unsqueeze(0), \
                                       size=(timg.shape[1], timg.shape[2]), mode='bilinear', align_corners=False).to(timg)
        bias_field_middle = F.interpolate(bias_field_ori_middle.unsqueeze(0).unsqueeze(0), \
                                          size=(timg.shape[1], timg.shape[2]), mode='bilinear', align_corners=False).to(timg)
        bias_field_high = F.interpolate(bias_field_ori_high.unsqueeze(0).unsqueeze(0), \
                                        size=(timg.shape[1], timg.shape[2]), mode='bilinear', align_corners=False).to(timg)

        # timg += torch.randn_like(timg) * sigma
        timg = timg * (bias_field_low.squeeze(0) ** self.real_low) * (bias_field_middle.squeeze(0) ** self.real_middle) \
            * (bias_field_high.squeeze(0) ** self.real_high) + torch.randn_like(timg) * real_sigma
        return simg, slab, timg, tlab


class PairedDataset(Dataset):
    """
    A dataset of paired images and labels.

    The dataset returns a three-tuple of:
        1) a label map to use for synth
        2) a real image
        3) a real label map
    If `split_synth_real` is False, the same label map is used for (1)
    and (3). Otherwise, the dataset is split in two, and the first half
    is used for synthesis and the other half for real examples.
    """

    def __init__(self, ndim, images, labels, split_synth_real=True,
                 subset=None, device=None):
        """

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        labels : [sequence of] str
            Input label maps: folder or file pattern or list of files.
        images : [sequence of] str
            Input images: folder or file pattern or list of files.
        split_synth_real : bool
            Do not use the same label map for real and synth examples
            in each batch.
        subset : slice or list[int]
            Only use a subset of the input files
        device : torch.device
            Device on which to load the data
        """
        self.ndim = ndim
        self.device = device
        self.split_synth_real = split_synth_real
        self.labels = np.asarray(folder2files(labels)[subset or slice(None)])
        self.images = np.asarray(folder2files(images)[subset or slice(None)])
        # NOTE: array[char] use less RAM that list[str] in multithreads
        assert len(self.labels) == len(self.images), \
               "Number of labels and images don't match"

    def __len__(self):
        n = len(self.images)
        if self.split_synth_real:
            n = n//2
        return n

    def __getitem__(self, idx):
        lab, img = str(self.labels[idx]), str(self.images[idx])

        lab = LoadTransform(ndim=self.ndim, dtype=torch.long, device=self.device)(lab)
        img = LoadTransform(ndim=self.ndim, dtype=torch.float32, device=self.device)(img)

        # mean = 0
        # var = 10
        # sigma = var ** 0.5
        # gaussian = torch.from_numpy(np.random.normal(mean, sigma, (img.shape[0], img.shape[1], img.shape[2])))
        # gaussian = gaussian.float()
        # img = img + gaussian

        if self.split_synth_real:
            slab = str(self.labels[len(self)+idx])
            slab = LoadTransform(ndim=self.ndim, dtype=torch.long, device=self.device)(slab)
            return slab, img, lab
        else:
            return lab, img, lab    


default_folder = '/autofs/cluster/vxmdata1/FS_Slim/proc/cleaned/OAS*/'
# default_folder = '/autofs/space/durian_001/users/xh999/learn2synth/data'


class PairedDataModule(pl.LightningDataModule):

    def __init__(self,
                 ndim: int,
                 images: Optional[Sequence[str]] = None,
                 labels: Optional[Sequence[str]] = None,
                 eval: Union[str, slice, List[int], int, float] = 0.2,
                 test: Union[str, slice, List[int], int, float] = 0.2,
                 preshuffle: bool = True,
                 shared: bool = False,
                 batch_size: int = 64,
                 shuffle: bool = False,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 ):
        """

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        images : sequence[str]
            List of images. By default, use `orig_talairach_slice` from
            vxmdata1, in folders that have `samseg_23_talairach_slice` labels.
        labels : sequence[str]
            List of label maps. By default, use `samseg_23_talairach_slice`
            from vxmdata1.
        eval : float
            Percentage of images to keep for evaluation
        test : float
            Percentage of images to keep for test
        preshuffle : bool
            Shuffle the image order once (otherwise, sort alphabetically)
        shared : bool
            Use the same image for the synth and real loss in each minibatch,
            Otherwise, use different images.
        batch_size : int
            Number of elements in a minibatch
        shuffle : bool
            Shuffle file order at each epoch
        num_workers : int
            Number of workers in the dataloader
        prefetch_factor : int
            Number of batches to load in advance
        """
        super().__init__()
        self.ndim = ndim
        
        # if labels is None:
        #     labels = sorted(glob(path.join(default_folder, '*seg.mgz')))
        # self.labels = list(labels)

        # if images is None:
        #     images = sorted(glob(path.join(default_folder, '*img.mgz')))

        if labels is None:
            # /autofs/cluster/vxmdata1/FS_Slim/proc/cleaned/OASIS_OAS1_0001_MR1/samseg_4_talairach_slice.mgz
            labels = sorted(glob(path.join(default_folder, 'samseg_23_talairach_slice.mgz')))
        self.labels = list(labels)
        if images is None:
            images = [path.join(path.dirname(label), 'norm_talairach_slice.mgz')
                      for label in self.labels]
            
        self.images = list(images)
        assert len(self.images) == len(self.labels), "Number of images and labels do not match"
        self.eval = parse_eval(eval)
        self.test = parse_eval(test)
        self.preshuffle = preshuffle
        self.shared = shared

        self.train_kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        self.eval_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    def setup(self, stage):
        images = self.images
        labels = self.labels
        if self.preshuffle:
            image_and_labels = list(zip(images, labels))
            shuffle(image_and_labels)
            images = [i for i, _ in image_and_labels]
            labels = [l for _, l in image_and_labels]

        slice_eval = self.eval
        if isinstance(slice_eval, float):
            slice_eval = int(math.ceil(len(images) * slice_eval))
        slice_test = self.test
        if isinstance(slice_test, float):
            slice_test = int(math.ceil(len(images) * slice_test))
        remaining_images, remaining_labels, \
        self.test_images, self.test_labels    \
            = splitset(images, labels, slice_test)
        self.train_images, self.train_labels, \
        self.eval_images, self.eval_labels    \
            = splitset(remaining_images, remaining_labels, slice_eval)

    def train_dataloader(self):
        train_dataset = PairedDataset(
            self.ndim, self.train_images, self.train_labels,
            split_synth_real=not self.shared)
        return DataLoader(train_dataset, **self.train_kwargs)

    def val_dataloader(self):
        eval_dataset = PairedDataset(
            self.ndim, self.eval_images, self.eval_labels,
            split_synth_real=not self.shared)
        return DataLoader(eval_dataset, **self.eval_kwargs)

    def test_dataloader(self):
        test_dataset = PairedDataset(
            self.ndim, self.test_images, self.test_labels,
            split_synth_real=not self.shared)
        return DataLoader(test_dataset, **self.eval_kwargs)


def parse_eval(eval):
    if not isinstance(eval, str):
        return eval
    if ':' in eval:
        eval = eval.split(':')
        eval = map(literal_eval, eval)
        eval = slice(*eval)
    else:
        try:
            eval = literal_eval(eval)
        except ValueError:
            pass
    return eval


def splitset(images, labels, eval):
    if isinstance(eval, float):
        eval = int(math.ceil(len(images) * eval))
    if isinstance(eval, int):
        eval = slice(-eval, None)
    if isinstance(eval, (slice, list, tuple)):
        eval_images = images[eval]
        eval_labels = labels[eval]
    else:
        eval_images = list(sorted(fnmatch.filter(images, eval)))
        eval_labels = list(sorted(fnmatch.filter(labels, eval)))
    images = list(sorted(filter(lambda x: x not in eval_images, images)))
    labels = list(sorted(filter(lambda x: x not in eval_labels, labels)))
    return images, labels, eval_images, eval_labels


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults({
            "checkpoint.monitor": "eval_loss",
            "checkpoint.save_last": True,
            "checkpoint.save_top_k": 5,
            "checkpoint.every_n_epochs": 10,
        })


if __name__ == '__main__':
    cli = CLI(Model, PairedDataModule)

