from learn2synth.scripts.train_gaussian import Model, PairedDataModule, ModelCheckpoint
import pytorch_lightning as pl
import os

from warnings import filterwarnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning
filterwarnings('ignore', category=PossibleUserWarning)
filterwarnings('ignore', 'torch.meshgrid')

outdir = f'/autofs/homes/007/yb947/links/experiments/learn2synth/shared/nolearn'
os.makedirs(outdir, exist_ok=True)

model = Model(
    ndim=2,
    seg_norm=None,
    classic=True,
    loss='logitmse',
    optimizer_options=dict(lr=1e-3),
)

checkpoints = ModelCheckpoint(
    monitor="eval_loss",
    save_last=True,
    save_top_k=5,
    every_n_epochs=10,
)

trainer = pl.Trainer(
    default_root_dir=outdir,
    callbacks=[checkpoints],
    accelerator='gpu',
    devices=[0],
    enable_checkpointing=True,
)

trainer.fit(model, datamodule=PairedDataModule(2, num_workers=0))

foo = 0