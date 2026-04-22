"""
MatBench runner for matbench_phonons task.

Two modes:
  scalar_regression (default, dos_num=1):
      For each fold, train the model from scratch on the fold's training structures
      and scalar phonon-peak targets, then predict on the test structures.

  dos_inference (dos_num=64):
      Load a pre-trained phonon-DOS model (--checkpoint), predict the full DOS
      for each test structure, and extract the last peak frequency using a
      configurable frequency grid (--max_phonon_freq).

Usage examples:
  # Train one model per fold (5-fold CV):
  python matbench_run.py --cfg configs/matbench_phonons.yaml --mode scalar_regression

  # Run inference with a pre-trained DOS model:
  python matbench_run.py --cfg configs/matbench_phonons.yaml --mode dos_inference \
      --checkpoint output/transformer/world_size1-STR/checkpoint_best.pth \
      --max_phonon_freq 1200.0

Results are saved to <outdir>/results.json.gz
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from matbench.bench import MatbenchBenchmark

from utils.structure_utils import find_last_peak, MAX_ATOMS
from datasets.matbench_dataset import MatbenchDataset
from utils.builder import ConfigBuilder
from utils.logger import get_logger
import utils.misc as utils


TASK_NAME = "matbench_phonons"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader(dataset: MatbenchDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def _build_model(builder: ConfigBuilder, logger):
    """Create a fresh model and wrap it for (single-GPU) inference/training."""
    model = builder.get_model()
    model_without_ddp = utils.DistributedParallel_Model(model, 0)
    for key in model_without_ddp.model:
        params = [p for p in model_without_ddp.model[key].parameters() if p.requires_grad]
        logger.info(f"params {key}: {sum(p.numel() for p in params):,}")
    return model_without_ddp


# ---------------------------------------------------------------------------
# Mode: scalar_regression
# ---------------------------------------------------------------------------

def run_scalar_regression(task, builder, cfg, outdir, logger):
    """
    For each fold:
      1. Train a fresh model on (structure, phonon_peak) training pairs.
      2. Predict phonon peaks on test structures.
      3. Record predictions.
    """
    num_workers  = cfg['dataloader'].get('num_workers', 2)
    batch_size   = cfg['trainer']['batch_size']
    test_batch   = cfg['trainer'].get('test_batch_size', 128)
    max_epoch    = cfg['trainer']['max_epoch']

    for fold in task.folds:
        logger.info(f"=== Fold {fold} / {task.folds[-1]} ===")
        fold_dir = os.path.join(outdir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        test_inputs = task.get_test_data(fold, include_target=False)

        train_ds = MatbenchDataset(list(train_inputs), targets=train_outputs)
        test_ds  = MatbenchDataset(list(test_inputs))

        train_loader = _make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers)
        test_loader  = _make_loader(test_ds,  test_batch, shuffle=False, num_workers=num_workers)

        model = _build_model(builder, logger)
        logger.info(f"Training fold {fold} for {max_epoch} epochs …")
        model.matbench_trainer(train_loader, max_epoch, checkpoint_savedir=fold_dir)

        preds = model.matbench_predict(test_loader)   # np.ndarray [N]
        predictions = pd.Series(preds.tolist(), index=test_inputs.index)

        task.record(fold, predictions)
        logger.info(f"Fold {fold} recorded. Sample predictions: {predictions.head()}")


# ---------------------------------------------------------------------------
# Mode: dos_inference
# ---------------------------------------------------------------------------

def run_dos_inference(task, builder, cfg, outdir, logger, checkpoint: str, max_phonon_freq: float):
    """
    Load a single pre-trained DOS model and apply it to every fold's test set.
    Phonon peak is extracted from predicted DOS using the configured frequency grid.
    """
    num_workers = cfg['dataloader'].get('num_workers', 2)
    test_batch  = cfg['trainer'].get('test_batch_size', 128)
    dos_num     = cfg['model']['params']['sub_model']['transformer']['dos_num']
    freq_grid   = np.linspace(0.0, max_phonon_freq, dos_num)

    # Build model and load checkpoint once (same weights for every fold)
    model = _build_model(builder, logger)
    model.load_checkpoint(checkpoint)
    logger.info(f"Loaded checkpoint: {checkpoint}  (dos_num={dos_num}, max_freq={max_phonon_freq})")

    for fold in task.folds:
        logger.info(f"=== Fold {fold} / {task.folds[-1]} ===")

        test_inputs = task.get_test_data(fold, include_target=False)
        test_ds     = MatbenchDataset(list(test_inputs))
        test_loader = _make_loader(test_ds, test_batch, shuffle=False, num_workers=num_workers)

        dos_preds = model.matbench_predict(test_loader)  # [N, dos_num]

        # Extract last peak from each DOS prediction
        peaks = np.array([
            find_last_peak(dos_preds[i], freq_grid)
            for i in range(len(dos_preds))
        ])
        predictions = pd.Series(peaks.tolist(), index=test_inputs.index)

        task.record(fold, predictions)
        logger.info(f"Fold {fold} recorded. Sample predictions: {predictions.head()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    utils.setup_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    logger = get_logger("matbench", args.outdir, 0, filename="matbench.log")

    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg['logger'] = logger
    builder = ConfigBuilder(**cfg)

    mb = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])
    task = getattr(mb, TASK_NAME)
    task.load()
    logger.info(f"Task '{TASK_NAME}' loaded: {task.metadata['n_samples']} samples, "
                f"{len(task.folds)} folds.")

    if args.mode == "scalar_regression":
        dos_num = cfg['model']['params']['sub_model']['transformer'].get('dos_num', 1)
        if dos_num != 1:
            logger.warning(
                f"scalar_regression mode expects dos_num=1 but config has dos_num={dos_num}. "
                "Training will produce a vector output — consider setting dos_num=1."
            )
        run_scalar_regression(task, builder, cfg, args.outdir, logger)

    elif args.mode == "dos_inference":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for dos_inference mode.")
        dos_num = cfg['model']['params']['sub_model']['transformer'].get('dos_num', 64)
        if dos_num == 1:
            raise ValueError(
                "dos_inference mode requires dos_num > 1. "
                "Use scalar_regression mode for dos_num=1."
            )
        run_dos_inference(
            task, builder, cfg, args.outdir, logger,
            checkpoint=args.checkpoint,
            max_phonon_freq=args.max_phonon_freq,
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}. Choose 'scalar_regression' or 'dos_inference'.")

    results_path = os.path.join(args.outdir, "results.json.gz")
    mb.to_file(results_path)
    logger.info(f"Benchmark saved → {results_path}")
    logger.info(mb.get_info())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MatBench phonons benchmark.")
    parser.add_argument(
        "--mode", type=str, default="scalar_regression",
        choices=["scalar_regression", "dos_inference"],
        help=(
            "scalar_regression: train dos_num=1 model per fold. "
            "dos_inference: load pre-trained DOS model and extract peak frequency."
        ),
    )
    parser.add_argument("--cfg", "-c",  type=str,  default=os.path.join("configs", "matbench_phonons.yaml"))
    parser.add_argument("--outdir",     type=str,  default="./output/matbench_phonons")
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to pre-trained checkpoint (required for dos_inference mode).",
    )
    parser.add_argument(
        "--max_phonon_freq", type=float, default=1200.0,
        help="Upper bound of phonon frequency grid in cm^-1 (dos_inference mode only).",
    )
    args = parser.parse_args()

    main(args)
