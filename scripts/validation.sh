OUTDIR="/autofs/space/durian_001/users/xh999/learn2synth/experiments/experiment_0099_test"

rm -rf $OUTDIR
mkdir $OUTDIR
python validation.py fit \
  --ckpt_path ../experiments/experiment_0099_1_1/lightning_logs/version_4186201/checkpoints/epoch=16399-step=402560.ckpt \
  --trainer.max_epochs 80000 \
  --model.ndim 2 \
  --data.ndim 2 \
  --trainer.default_root_dir $OUTDIR \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --data.num_workers 4 \
  --model.classic false \
  --model.loss logitmse \
  --model.optimizer_options "{'lr': 0.0001}"
