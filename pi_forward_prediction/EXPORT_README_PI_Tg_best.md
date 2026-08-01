# PI Tg Best Model Export Package

## Experiment

- Experiment name: `kfold_m128_head15`
- Training pool: 1,723 PI samples with P1/m128 descriptor features
- Five-fold validation: mean RMSE about 25.39 deg C, R2 about 0.926
- Metrics file: `experiments/results/p26_run_kfold_m128_head15.json`

## Inference Artifacts

Load order:

1. MLM pretraining checkpoint: `ckpt/pretrain.pt/`
2. Contrastive RoBERTa backbone: `ckpt/experiments/full/contrastive_roberta/`
3. Downstream Tg checkpoint: `ckpt/experiments/p26/kfold_m128_head15/PI_Tg_best_model.pt`

The downstream config is:

```text
experiments/config/p26_runs/kfold_m128_head15.yaml
```

It enables descriptor fusion. Descriptor column names are stored in the YAML `descriptor_cols` field.

## Optional Fine-Tuning Reproduction

```bash
cd pi_forward_prediction
python Downstream.py --config experiments/config/p26_runs/kfold_m128_head15.yaml
```

## Not Included

- The million-scale unlabeled `pretrain.csv` corpus
- Per-fold `fold_*_best.pt` checkpoints
- The intermediate `PI_Tg_train.pt` checkpoint

These files are not required for packaged inference.
