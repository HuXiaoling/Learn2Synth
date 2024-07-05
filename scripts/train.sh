#OUTDIR=~/links/experiments/learn2synth/shared/nolearn

ALPHA=12
OUTDIR="/autofs/space/durian_001/users/xh999/learn2synth/experiment_${ALPHA}"

# --ckpt_path ../experiment_5/lightning_logs/version_0/checkpoints/epoch=9489-step=322660.ckpt \

rm -rf $OUTDIR
mkdir $OUTDIR
python train_gaussian.py fit \
  --ckpt_path ../experiment_8/lightning_logs/version_0/checkpoints/last.ckpt \
  --trainer.max_epochs 20000 \
  --model.ndim 2 \
  --data.ndim 2 \
  --trainer.default_root_dir $OUTDIR \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --data.num_workers 4 \
  --model.classic false \
  --model.alpha $ALPHA \
  --model.loss logitmse \
  --model.optimizer_options "{'lr': 0.0001}"
