#OUTDIR=~/links/experiments/learn2synth/shared/nolearn

ALPHA=1
try=1
OUTDIR="/autofs/space/durian_001/users/xh999/learn2synth/experiments/experiment_noise_${ALPHA}_${try}"

# --ckpt_path ../experiment_5/lightning_logs/version_0/checkpoints/epoch=9489-step=322660.ckpt \

rm -rf $OUTDIR
mkdir $OUTDIR
python train_noise.py fit \
  --trainer.max_epochs 60000 \
  --model.ndim 2 \
  --data.ndim 2 \
  --trainer.default_root_dir $OUTDIR \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --data.num_workers 4 \
  --model.classic true \
  --model.real_sigma_min 0 \
  --model.real_sigma_max 0 \
  --model.alpha $ALPHA \
  --model.loss logitmse \
  --model.optimizer_options "{'lr': 0.001}"
