import argparse
import datetime
import os
import time

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import (
    CSVLogger,
    MLFlowLogger,
    TensorBoardLogger,
    WandbLogger,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from utils.util import instantiate_from_config


def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        v = v.lower()
        if v in ("yes", "true", "t", "y", "1"):
            return True
        if v in ("no", "false", "f", "n", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="Path(s) to base configs.",
        default="stage1/configs/lora_config_benchmark_tasks.yaml",
    )
    parser.add_argument("-n", "--name", type=str, default="run", help="Run name prefix.")
    parser.add_argument("-r", "--resume", type=str, default="", help="Path to ckpt.")
    parser.add_argument("-l1", "--logdir", type=str, default="logs", help="Log root dir.")
    parser.add_argument("-t", "--train", type=str2bool, default=False, help="Enable training.")
    parser.add_argument("--no-test", type=str2bool, default=False, help="Disable validation after fit.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "csv", "wandb", "mlflow", "none"],
        help="Logger backend.",
    )
    return parser


class CleanEmptyCheckpointFolders(Callback):
    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        ckpt_dir = trainer.checkpoint_callback.dirpath
        if ckpt_dir and os.path.exists(ckpt_dir):
            for item in os.listdir(ckpt_dir):
                full_path = os.path.join(ckpt_dir, item)
                if os.path.isdir(full_path) and not os.listdir(full_path):
                    os.rmdir(full_path)


@rank_zero_only
def save_config(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    OmegaConf.save(cfg, path)
    print(f"Saved config to {path}")


class LossScheduleCB(Callback):

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        loss_obj = getattr(pl_module, "loss_fn", None) or getattr(pl_module, "loss", None)
        if loss_obj is None:
            return
        total = getattr(trainer, "estimated_stepping_batches", None) or trainer.max_steps or 1
        step = trainer.global_step
        if hasattr(loss_obj, "set_step"):
            loss_obj.set_step(step, total)


def main():
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if isinstance(opt.base, str):
        configs = [OmegaConf.load(opt.base)]
    else:
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_name = f"{opt.name}_{now}"
    log_dir = os.path.join(opt.logdir, "stage1", run_name)
    config.name = run_name

    pl.seed_everything(opt.seed)
    save_config(config, os.path.join(log_dir, "config.yaml"))

    model = instantiate_from_config(config.model)
    datamodule = instantiate_from_config(config.data)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    if opt.logger == "tensorboard":
        logger = TensorBoardLogger(save_dir=log_dir, name="", default_hp_metric=False)
    elif opt.logger == "csv":
        logger = CSVLogger(save_dir=log_dir, name="")
    elif opt.logger == "wandb":
        logger = WandbLogger(name=config.name, save_dir=log_dir, project="d2nwg")
    elif opt.logger == "mlflow":
        logger = MLFlowLogger(experiment_name="d2nwg_exp", tracking_uri=os.path.join(log_dir, "mlruns"))
    else:
        logger = False

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    ckpt_cfg = config.get("checkpoint", {})
    save_top_k = ckpt_cfg.get("save_top_k", 5)
    filename = ckpt_cfg.get("filename", ckpt_cfg.get("file_name", "epoch{epoch:06d}-aeloss{train/aeloss:.8f}"))

    checkpoint_callback_topk_only = ModelCheckpoint(
        monitor="train/aeloss",
        save_top_k=save_top_k,
        mode="min",
        filename=filename,
        dirpath=ckpt_dir,
        every_n_epochs=1,
    )

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true",
        accelerator=config.get("trainer", {}).get("accelerator", "gpu"),
        devices=config.get("trainer", {}).get("devices", -1),
        max_epochs=config.get("trainer", {}).get("max_epochs", 1000),
        min_epochs=config.get("trainer", {}).get("min_epochs", 100),
        log_every_n_steps=config.get("trainer", {}).get("log_every_n_steps", 1),
        default_root_dir=log_dir,
        logger=logger,
        callbacks=[
            checkpoint_callback_topk_only,
            CleanEmptyCheckpointFolders(),
            LossScheduleCB(),
        ],
    )

    start_time = time.time()

    if opt.train:
        if opt.resume:
            print(f"Resuming from: {opt.resume}")
            trainer.fit(model, datamodule=datamodule, ckpt_path=opt.resume)
        else:
            trainer.fit(model, datamodule=datamodule)

    if not opt.no_test:
        print("Running validation...")
        trainer.validate(model, datamodule=datamodule)

    elapsed = time.time() - start_time
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f"Total runtime: {h}h {m}m {s}s ({elapsed:.2f} seconds)")


if __name__ == "__main__":
    main()


'''
    Usage:
    eg.
    # ============ flanv2 ============
    CUDA_VISIBLE_DEVICES=0 python main_stage1.py \
        -b stage1/configs/lora_config_flanv2_sub.yaml \
        -n lora_config_flanv2_sub \
        --train true \
        --logger tensorboard

    CUDA_VISIBLE_DEVICES=0 python main_stage1.py \
        -b stage1/configs/lora_config_flanv2_zero_shot_ex.yaml \
        -n lora_config_flanv2_zero_shot \
        --train true \
        --logger tensorboard

    # ============ bench_tasks ============
    CUDA_VISIBLE_DEVICES=0 python main_stage1.py \
        -b stage1/configs/lora_config_bench_tasks.yaml \
        -n lora_config_bench_tasks \
        --train true \
        --logger tensorboard

'''