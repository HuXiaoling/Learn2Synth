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
    
        # ================================================================
        # 1. SYNTHETIC PASS (updates segmentation network parameters φ)
        # ================================================================
        # We call clone_module(self.segnet) instead of using self.segnet directly.
        # Why? Because we are about to update φ in-place via optim_seg.step().
        # If we forward with self.segnet directly, the in-place parameter update
        # would break the computation graph, and we could not backpropagate through
        # the update step when computing hypergradients in the real pass.
        #
        # clone_module works as follows:
        # - It creates a shallow copy of the module structure.
        # - Each parameter is replaced with param.clone(), which is NOT a leaf
        #   variable but still has autograd dependency on the original parameter.
        #   Thus gradients flow back to the original φ.
        #
        # Effectively, clone_module gives us a "clean copy" to do forward passes,
        # while gradients still accumulate on the real self.segnet parameters.
        self.train()
        synth_pred = clone_module(self.segnet)(synth_image)
    
        # Synthetic loss: L_synth = ℓ(f_φ(A_θ(x_synth)), y_synth)
        synth_loss = self.loss(synth_pred, synth_ref)
    
        # Backward w/ create_graph=True:
        # This ensures that the parameter update step (optim_seg.step)
        # is itself recorded in the computation graph.
        # Without create_graph, φ* would just be a detached tensor,
        # and we could not differentiate through the update wrt θ later.
        if self.backward:
            self.backward(synth_loss, inputs=list(optim_seg.parameters()), create_graph=True)
        else:
            synth_loss.backward(inputs=list(optim_seg.parameters()), create_graph=True)
    
        # Perform the parameter update:
        # φ* = φ − η ∇_φ L_synth(A_θ(x_synth), y_synth)
        # Importantly, because of create_graph=True, φ* "remembers"
        # its dependence on θ through A_θ. This establishes the link
        # between θ and any future real loss that uses φ*.
        optim_seg.step()
    
        # ================================================================
        # 2. REAL PASS (updates augmentation network parameters θ)
        # ================================================================
        # Now we evaluate the segmentation network on real data.
        # Note: φ* (the updated parameters) are implicitly used here,
        # so the real loss depends on θ indirectly via φ*.
        # No need to clone this time, since we do not want to create
        # a new graph or disturb normalization statistics.
        self.eval()
        real_pred = self.segnet(real_image)
    
        # Real loss: L_real = ℓ(f_{φ*}(x_real), y_real)
        real_loss = self.loss(real_pred, real_ref)
    
        # Backward on real_loss wrt θ:
        # Even though x_real never passes through A_θ,
        # the chain rule applies:
        #   ∂L_real/∂θ = (∂L_real/∂φ*) · (∂φ*/∂θ)
        # This is possible because φ* was computed in the synthetic pass
        # with create_graph=True, so autograd knows how φ* depends on θ.
        #
        # In other words: φ* "carries the imprint" of θ,
        # and L_real leverages that to compute a hypergradient.
        if self.backward:
            self.backward(real_loss.mul(self.alpha), inputs=list(optim_synth.parameters()))
        else:
            real_loss.mul(self.alpha).backward(inputs=list(optim_synth.parameters()))
    
        # Update augmentation parameters θ
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

