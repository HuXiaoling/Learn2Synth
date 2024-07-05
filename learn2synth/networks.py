from . import modules, utils
from torch import nn


def _init_from_defaults(self, **kwargs):
    for key, value in self.defaults.__dict__.items():
        if key[0] != '_':
            kwargs.setdefault(key, value)
    for key, value in kwargs.items():
        setattr(self, key, value)


class SegNet(nn.Sequential):
    """
    A generic segmentation network that works with any backbone
    """

    def __init__(self, ndim, in_channels, out_channels,
                 kernel_size=3, activation='Softmax',
                 backbone='UNet', kwargs_backbone=None):
        """

        Parameters
        ----------
        ndim : {2, 3}
            Number of spatial dimensions
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output classes
        kernel_size : int, default=3
            Kernel size of the initial feature extraction layer
        activation : str or None, default='softmax'
            Final activation function
        backbone : {'UNet', 'MeshNet', 'ATrousNet'} or Module, default='UNet'
            Generic backbone module. Can be already instantiated.
        kwargs_backbone : dict, optional
            Parameters of the backbone (if backbone is not pre-instantiated)
        """
        if isinstance(backbone, str):
            backbone_kls = globals()[backbone]
            backbone = backbone_kls(ndim, **(kwargs_backbone or {}))
        if activation and activation.lower() == 'softmax':
            activation = nn.Softmax(1)
        feat = modules.ConvBlock(ndim,
                                 in_channels, backbone.in_channels,
                                 kernel_size=kernel_size,
                                 activation=None)
        pred = modules.ConvBlock(ndim,
                                 backbone.out_channels, out_channels,
                                 kernel_size=1,
                                 activation=activation)
        super().__init__(feat, backbone, pred)


class UNet(nn.Module):
    """A highly parameterized U-Net (encoder-decoder + skip connections)

    conv ------------------------------------------(+)-> conv
         -down-> conv ---------------(+)-> conv -> up
                     -down-> conv -> up

    Reference
    ---------
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    MICCAI (2015)
    https://arxiv.org/abs/1505.04597
    """

    class defaults:
        nb_levels = 6            # number of levels
        nb_features = (16, 24, 32, 48, 64, 96, 128, 192, 256, 320)
        nb_conv = 2              # number of convolutions per level
        kernel_size = 3          # kernel size
        activation = 'ReLU'      # activation function
        norm = 'instance'        # 'batch', 'instance', 'layer', None
        dropout = 0              # dropout probability
        residual = False         # use residual connections throughout
        factor = 2               # change resolution by this factor
        use_strides = False      # use strided conv instead of linear resize
        order = 'cand'           # c[onv], a[ctivation], n[orm], d[ropout]
        combine = 'cat'          # 'cat', 'add'

    def _feat_block(self, i, o=None):
        opt = dict(activation=None, kernel_size=self.kernel_size,
                   order=self.order, residual=self.residual)
        return modules.ConvBlock(self.ndim, i, o, **opt)

    def _conv_block(self, i, o=None):
        opt = dict(activation=self.activation, kernel_size=self.kernel_size,
                   order=self.order, nb_conv=self.nb_conv, norm=self.norm,
                   residual=self.residual, dropout=self.dropout)
        return modules.ConvGroup(self.ndim, i, o, **opt)

    def _down_block(self, i, o=None):
        if self.use_strides:
            opt = dict(activation=self.activation, strides=self.factor,
                       order=self.order, kernel_size=self.factor,
                       norm=self.norm, dropout=self.dropout)
            return modules.StridedConvBlockDown(self.ndim, i, o, **opt)
        else:
            opt = dict(activation=self.activation, factor=self.factor,
                       order=self.order, kernel_size=1,
                       norm=self.norm, dropout=self.dropout)
            return modules.ConvBlockDown(self.ndim, i, o, **opt)

    def _up_block(self, i, o=None):
        if self.use_strides:
            opt = dict(activation=self.activation, strides=self.factor,
                       order=self.order, kernel_size=self.factor,
                       norm=self.norm, dropout=self.dropout,
                       combine=self.combine)
            return modules.StridedConvBlockUp(self.ndim, i, o, **opt)
        else:
            opt = dict(activation=self.activation, factor=self.factor,
                       order=self.order, kernel_size=1,
                       norm=self.norm, dropout=self.dropout,
                       combine=self.combine)
            return modules.ConvBlockUp(self.ndim, i, o, **opt)

    def __init__(self, ndim, **kwargs):
        super().__init__()
        _init_from_defaults(self, **kwargs)
        self.ndim = ndim
        self.nb_features = utils.ensure_list(self.nb_features, self.nb_levels)
        self.in_channels = self.out_channels = self.nb_features[0]

        # encoder
        i, o = self.nb_features[0], self.nb_features[0]
        self.encoder = [self._conv_block(i, o)]
        for n in range(1, len(self.nb_features)-1):
            i, o = self.nb_features[n-1], self.nb_features[n]
            self.encoder += [modules.EncoderBlock(self._down_block(i, o),
                                                  self._conv_block(o))]
        if self.nb_levels > 1:
            i, o = self.nb_features[-2], self.nb_features[-1]
            self.encoder += [self._down_block(i, o)]
        self.encoder = nn.Sequential(*self.encoder)

        # decoder
        self.decoder = []
        for n in range(len(self.nb_features)-1):
            i, o = self.nb_features[-n-1], self.nb_features[-n-2]
            m = i
            if self.combine == 'cat' and n > 0:
                i *= 2
            self.decoder += [modules.DecoderBlock(self._conv_block(i, m),
                                                  self._up_block(m, o))]
        i, o = self.nb_features[0], self.nb_features[0]
        if self.nb_levels > 1 and self.combine == 'cat':
            i *= 2
        self.decoder += [self._conv_block(i, o)]
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        # check shape
        nb_levels = len(self.encoder)
        if any(s < 2**nb_levels for s in x.shape[2:]):
            raise ValueError(f'UNet with {nb_levels} levels requires input '
                             f'shape larger or equal to {2**nb_levels}, but '
                             f'got {list(x.shape[2:])}')

        # compute downstream pyramid
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        # compute upstream pyramid
        x = skips.pop(-1)
        for n in range(len(self.decoder)-1):
            x = self.decoder[n].conv(x)
            x = self.decoder[n].up(x, skips.pop(-1))

        # highest resolution
        x = self.decoder[-1](x)

        return x


class MeshNet(nn.Sequential):
    """A stack of dilated convolutions

    conv(rate=1) -> conv(rate=2) -> conv(rate=4) -> conv(rate=2) -> conv(rate=1)

    Reference
    ---------
    "Multi-Scale Context Aggregation by Dilated Convolutions"
    Fisher Yu, Vladlen Koltun
    ICLR (2016)
    https://arxiv.org/abs/1511.07122

    "End-to-end learning of brain tissue segmentation from imperfect labeling"
    Alex Fedorov, Jeremy Johnson, Eswar Damaraju, Alexei Ozerin, Vince Calhoun, Sergey Plis
    IJCNN (2017)
    https://arxiv.org/abs/1612.00940
    """

    class defaults:
        nb_levels = 6           # max dilation = factor ** (nb_levels-1)
        nb_features = 24        # number of features per level. Can be a list.
        nb_conv = 2             # Number of convolutions per layer/
        kernel_size = 3
        activation = 'ReLU'
        norm = 'instance'       # 'batch', 'instance', 'layer', None
        dropout = 0             # dropout probability
        residual = False        # Use residual connections in conv blocks
        factor = 2              # Dilation factor per level.
        order = 'cand'          # c[onv], a[ctivation], n[orm], d[ropout]

    def _feat_block(self, i, o=None):
        opt = dict(activation=None, kernel_size=self.kernel_size,
                   order=self.order, residual=self.residual)
        return modules.ConvBlock(self.ndim, i, o, **opt)

    def _conv_block(self, d, i, o=None):
        opt = dict(activation=self.activation, kernel_size=self.kernel_size,
                   order=self.order, nb_conv=self.nb_conv, norm=self.norm,
                   residual=self.residual, dropout=self.dropout,
                   dilation=d)
        return modules.ConvGroup(self.ndim, i, o, **opt)

    def __init__(self, ndim, **kwargs):
        _init_from_defaults(self, **kwargs)
        self.ndim = ndim
        self.nb_features = utils.ensure_list(self.nb_features, self.nb_levels)
        self.in_channels = self.out_channels = self.nb_features[0]

        layers = []
        # encoder
        for i in range(self.nb_levels):
            layers += [self._conv_block(self.factor**i, self.nb_features[i])]
        # decoder
        for i in reversed(range(self.nb_levels-1)):
            layers += [self._conv_block(self.factor**i, self.nb_features[i])]
        super().__init__(*layers)

    def forward(self, x):
        if (self.residual and
                all([f == self.nb_features[0] for f in self.nb_features])):
            for layer in self:
                skip = x
                x = layer(x)
                x += skip
            return x
        else:
            return super().forward(x)


class ATrousNet(nn.Sequential):
    """Parallel dilated convolutions

    conv(rate=1) → conv(rate=1)  ⊕  → conv(rate=1)  ⊕  → conv(rate=1)  ⊕ → conv(rate=1)
                 ↳ conv(rate=2) -^  ↳ conv(rate=2) -^  ↳ conv(rate=2) -^
                                    ↳ conv(rate=4) -^

    Reference
    ---------
    "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
    Atrous Convolution, and Fully Connected CRFs"
    Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille
    TPAMI (2017)
    https://arxiv.org/abs/1606.00915
    """

    class defaults:
        nb_levels = 6           # max dilation = factor ** (nb_levels-1)
        nb_features = 24        # number of features per level. Can be a list.
        nb_conv = 2             # Number of convolutions per layer/
        kernel_size = 3
        activation = 'ReLU'
        norm = 'instance'       # 'batch', 'instance', 'layer', None
        dropout = 0             # dropout probability
        residual = False        # Use residual connections throughout
        factor = 2              # Dilation factor per level.
        order = 'cand'          # c[onv], a[ctivation], n[orm], d[ropout]

    def _feat_block(self, i, o=None):
        opt = dict(activation=None, kernel_size=self.kernel_size,
                   order=self.order, residual=self.residual)
        return modules.ConvBlock(self.ndim, i, o, **opt)

    def _conv_block(self, d, i, o=None):
        opt = dict(activation=self.activation, kernel_size=self.kernel_size,
                   order=self.order, nb_conv=self.nb_conv, norm=self.norm,
                   residual=self.residual, dropout=self.dropout,
                   dilation=d)
        return modules.ConvGroup(self.ndim, i, o, **opt)

    def __init__(self, ndim, **kwargs):
        _init_from_defaults(self, **kwargs)
        self.ndim = ndim
        self.nb_features = utils.ensure_list(self.nb_features, self.nb_levels)
        self.in_channels = self.out_channels = self.nb_features[0]

        layers = []
        # encoder
        for i in range(self.nb_levels):
            sublayers = []
            for j in range(i+1):
                sublayers += [self._conv_block(self.factor**j, self.nb_features[j])]
            layers += [nn.ModuleList(sublayers)]
        # decoder
        for i in reversed(range(self.nb_levels-1)):
            for j in range(i+1):
                sublayers += [self._conv_block(self.factor**j, self.nb_features[j])]
            layers += [nn.ModuleList(sublayers)]
        super().__init__(*layers)

    def forward(self, x):
        do_residual = (self.residual and all([f == self.nb_features[0]
                                              for f in self.nb_features]))
        for layer in self:
            skip, x = x, None
            for sublayer in layer:
                if x is None:
                    x = sublayer(skip)
                else:
                    x += sublayer(skip)
            if do_residual:
                x += skip
        return x


class DeepVesselBackbone(nn.Sequential):
    """
    conv(size=3) -> conv(size=5) -> conv(size=5) -> conv(size=3)

    References
    ----------
    "DeepVesselNet: Vessel Segmentation, Centerline Prediction,
    and Bifurcation Detection in 3-D Angiographic Volumes"
    Tetteh et al
    Front. Neurosci. (2020)
    https://www.frontiersin.org/articles/10.3389/fnins.2020.592352/full
    """

    class defaults:
        nb_levels = 3
        nb_features = (5, 10, 20, 50)   # number of features per level.
        nb_conv = 1                     # Number of convolutions per layer
        kernel_size = (3, 5, 5, 3)      # Kernel-size per layer
        activation = 'Tanh'
        norm = None                     # 'batch', 'instance', 'layer', None
        dropout = 0                     # dropout probability
        residual = False                # Use residual connections throughout
        order = 'cand'                  # c[onv], a[ctivation], n[orm], d[ropout]
        separable = 'cross'             # bool or 'cross'

    def _conv_block(self, i, o=None):
        opt = dict(activation=self.activation, kernel_size=self.kernel_size,
                   order=self.order, nb_conv=self.nb_conv, norm=self.norm,
                   residual=self.residual, dropout=self.dropout,
                   separable=self.separable)
        return modules.ConvGroup(self.ndim, i, o, **opt)

    def __init__(self, ndim, **kwargs):
        _init_from_defaults(self, **kwargs)
        self.ndim = ndim
        self.nb_features = utils.ensure_list(self.nb_features, self.nb_levels+1)
        self.in_channels = self.nb_features[0]
        self.out_channels = self.nb_features[-1]

        layers = []
        for i in range(self.nb_levels):
            layers += [self._conv_block(self.nb_features[i], self.nb_features[i+1])]
        super().__init__(*layers)


class DeepVesselNet(nn.Sequential):
    """
    References
    ----------
    "DeepVesselNet: Vessel Segmentation, Centerline Prediction,
    and Bifurcation Detection in 3-D Angiographic Volumes"
    Tetteh et al
    Front. Neurosci. (2020)
    https://www.frontiersin.org/articles/10.3389/fnins.2020.592352/full
    """

    def __init__(self, ndim, in_channels, out_channels,
                 activation='Softmax', kwargs_backbone=None):
        """
        Parameters
        ----------
        ndim : {2, 3}
            Number of spatial dimensions
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output classes
        activation : str, default='softmax'
            Final activation function
        kwargs_backbone : dict, optional
            Parameters of the DeepVessel backbone
        """
        if activation.lower() == 'softmax':
            activation = nn.Softmax(1)
        nb_features = kwargs_backbone.get('nb_features', DeepVesselBackbone.defaults.nb_features)
        nb_features = [in_channels, *utils.ensure_list(nb_features)]
        backbone = DeepVesselBackbone(ndim, nb_features=nb_features,
                                      **(kwargs_backbone or {}))
        pred = modules.ConvBlock(ndim,
                                 backbone.out_channels, out_channels,
                                 kernel_size=1,
                                 activation=activation)
        super().__init__(backbone, pred)
