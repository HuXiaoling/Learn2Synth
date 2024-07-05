import torch
from torch import nn
from .modules import clone as clone_module


class LearnableSynthSeg(nn.Module):

    def __init__(self, segnet, synth, synthnet, loss, alpha=1., residual=True, noise=False):
        """

        Parameters
        ----------
        segnet : nn.Module
            Segmentation network:  (B, 1, *S) image -> (B, K, *S) prob
        synth : Transform, optional
            Synthesis block, without learnable weights:
            (B, 1, *S) label map -> [(B, 1, *S) image, (B, 1, *S) ref]
        synthnet : nn.Module, optional
            Learnable synthesis network:
            (B, 1, *S) image ->  (B, 1, *S) image
        loss : nn.Module
            Segmentation loss: [(B, 1, *S) pred, (B, 1, *S) ref] -> scalar
        alpha : float
            Multiplicative factor for the real loss
        residual : bool
            Whether the synthnet is residual or not
        noise : bool
            Whether to provide an additional channel of random noise as
            input to the synthnet (ddpm-like)
        """
        super().__init__()
        self.segnet = segnet
        self.synth = synth
        self.synthnet = synthnet
        self.loss = loss
        self.residual = residual
        self.noise = noise
        self.alpha = alpha
        self.optim_seg = None
        self.optim_synth = None
        self.backward = None
        self.optimizers = None

    def forward(self, x):
        return self.segnet(x)

    def configure_optimizers(self, optim_seg, optim_synth=None):
        optim_synth = optim_synth or optim_seg
        if callable(optim_seg):
            optim_seg = optim_seg(self.segnet.parameters())
        if callable(optim_synth):
            optim_synth = optim_synth(self.synthnet.parameters())
        def optimizers():
            return optim_seg, optim_synth
        self.optimizers = optimizers
        return optimizers()

    def set_optimizers(self, optimizers):
        self.optimizers = optimizers

    def set_backward(self, backward):
        self.backward = backward

    def reset_backward(self):
        self.backward = None

    def synthplus(self, img):
        if self.synthnet:
            inp = img
            if self.noise:
                inp = torch.cat([inp, torch.randn_like(img)], dim=1)
            if self.residual:
                img = self.synthnet(inp).add_(img)
            else:
                img = self.synthnet(inp)
        return img

    def synth_and_train_step(self, label, real_image, real_ref):
        self.train()
        synth_image, synth_ref, real_image, real_ref = self.synth(label, real_image, real_ref)
        synth_image = self.synthplus(synth_image)
        return self.train_step(synth_image, synth_ref, real_image, real_ref)

    def synth_and_eval_step(self, label, real_image, real_ref):
        self.eval()
        synth_image, synth_ref, real_image, real_ref = self.synth(label, real_image, real_ref)
        synth_image_plus = self.synthplus(synth_image)
        return self.eval_step(synth_image_plus, synth_image, synth_ref, real_image, real_ref)

    def synth_and_eval_for_plot(self, label, real_image, real_ref):
        self.eval()
        synth_image, synth_ref, real_image, real_ref = self.synth(label, real_image, real_ref)
        synth_image_plus = self.synthplus(synth_image)
        return *self.eval_for_plot(synth_image_plus, synth_image, synth_ref, real_image, real_ref), synth_image_plus, synth_image, synth_ref, real_image, real_ref

    def train_step(self, synth_image, synth_ref, real_image, real_ref):
        optim_seg, optim_synth = self.optimizers()

        optim_seg.zero_grad()
        optim_synth.zero_grad()

        # synth forward
        # we must call clone_module so that a copy of all the weights
        # is performed before the in-place update.
        # Otherwise, we could not backpropagate through the weights
        # after the update.
        self.train()
        synth_pred = clone_module(self.segnet)(synth_image)
        synth_loss = self.loss(synth_pred, synth_ref)
        if self.backward:
            self.backward(synth_loss, inputs=list(optim_seg.parameters()), create_graph=True)
        else:
            synth_loss.backward(inputs=list(optim_seg.parameters()), create_graph=True)
        optim_seg.step()

        # real forward
        # no need to clone here
        # eval mode because we do not want to accumulate norm stats
        self.eval()
        real_pred = self.segnet(real_image)
        real_loss = self.loss(real_pred, real_ref)
        if self.backward:
            self.backward(real_loss.mul(self.alpha), inputs=list(optim_synth.parameters()))
        else:
            real_loss.mul(self.alpha).backward(inputs=list(optim_synth.parameters()))
        optim_synth.step()

        return synth_loss, real_loss

    def eval_step(self, synth_image_plus, synth_image, synth_ref, real_image, real_ref):
        self.eval()
        with torch.no_grad():

            # synth forward
            synth_pred = self.segnet(synth_image)
            synth_loss = self.loss(synth_pred, synth_ref)

            # synth plus forward
            synth_plus_pred = self.segnet(synth_image_plus)
            synth_plus_loss = self.loss(synth_plus_pred, synth_ref)

            # real forward
            real_pred = self.segnet(real_image)
            real_loss = self.loss(real_pred, real_ref)

        return synth_plus_loss, synth_loss, real_loss

    def eval_for_plot(self, synth_image_plus, synth_image, synth_ref, real_image, real_ref):
        self.eval()
        with torch.no_grad():

            # synth forward
            synth_pred = self.segnet(synth_image)
            synth_loss = self.loss(synth_pred, synth_ref)

            # synth plus forward
            synth_plus_pred = self.segnet(synth_image_plus)
            synth_plus_loss = self.loss(synth_plus_pred, synth_ref)

            # real forward
            real_pred = self.segnet(real_image)
            real_loss = self.loss(real_pred, real_ref)

        return synth_plus_loss, synth_loss, real_loss, synth_plus_pred, synth_pred, real_pred


class SynthSeg(nn.Module):
    """A SynthSeg network, except that we evaluate it on real data as well"""

    def __init__(self, segnet, synth, loss):
        """

        Parameters
        ----------
        segnet : nn.Module
            Segmentation network:  (B, 1, *S) image -> (B, K, *S) prob
        synth : Transform, optional
            Synthesis block, without learnable weights:
            (B, 1, *S) label map -> [(B, 1, *S) image, (B, 1, *S) ref]
        loss : nn.Module
            Segmentation loss: [(B, 1, *S) pred, (B, 1, *S) ref] -> scalar
        """
        super().__init__()
        self.segnet = segnet
        self.synth = synth
        self.loss = loss
        self.optim = None
        self.backward = None
        self.optimizers = None

    def forward(self, x):
        return self.segnet(x)

    def synthesize(self, label):
        img, ref = self.synth(label)
        return img, ref

    def configure_optimizers(self, optim):
        if callable(optim):
            optim = optim(self.segnet.parameters())
        def optimizers():
            return optim
        self.optimizers = optimizers
        return optimizers()

    def set_optimizers(self, optimizers):
        self.optimizers = optimizers

    def set_backward(self, backward):
        self.backward = backward

    def reset_backward(self):
        self.backward = None

    def synth_and_train_step(self, label, real_image, real_ref):
        synth_image, synth_ref, real_image, real_ref = self.synth(label, real_image, real_ref)
        return self.train_step(synth_image, synth_ref, real_image, real_ref)

    def synth_and_eval_step(self, label, real_image, real_ref):
        synth_image, synth_ref, real_image, real_ref = self.synth(label, real_image, real_ref)
        return self.eval_step(synth_image, synth_ref, real_image, real_ref)

    def synth_and_eval_for_plot(self, label, real_image, real_ref):
        self.eval()
        synth_image, synth_ref, real_image, real_ref = self.synth(label, real_image, real_ref)
        return *self.eval_for_plot(synth_image, synth_ref, real_image, real_ref), synth_image, synth_ref, real_image, real_ref

    def train_step(self, synth_image, synth_ref, real_image, real_ref):
        optim = self.optimizers()
        optim.zero_grad()

        # synth forward
        self.train()
        synth_pred = self.segnet(synth_image)
        synth_loss = self.loss(synth_pred, synth_ref)
        if self.backward:
            self.backward(synth_loss)
        else:
            synth_loss.backward()
        optim.step()

        self.eval()
        with torch.no_grad():
            # real forward
            real_pred = self.segnet(real_image)
            real_loss = self.loss(real_pred, real_ref)

        return synth_loss, real_loss

    def eval_step(self, synth_image, synth_ref, real_image, real_ref):
        self.eval()
        with torch.no_grad():

            # synth forward
            synth_pred = self.segnet(synth_image)
            synth_loss = self.loss(synth_pred, synth_ref)

            # real forward
            real_pred = self.segnet(real_image)
            real_loss = self.loss(real_pred, real_ref)

        return synth_loss, real_loss

    def eval_for_plot(self, synth_image, synth_ref, real_image, real_ref):
        self.eval()
        with torch.no_grad():

            # synth forward
            synth_pred = self.segnet(synth_image)
            synth_loss = self.loss(synth_pred, synth_ref)

            # real forward
            real_pred = self.segnet(real_image)
            real_loss = self.loss(real_pred, real_ref)

        return synth_loss, real_loss, synth_pred, real_pred

