#!/usr/bin/env python3

from pytorch_lightning import cli_lightning_logo
from pytorch_lightning.cli import LightningCLI

from transferLearning import BEATsTransferLearningModel
from DataModule import BirdDataModule
from callbacks import MilestonesFinetuning


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(MilestonesFinetuning, "finetuning")
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("finetuning.milestones", "model.milestones")
        parser.set_defaults(
            {
                "trainer.max_epochs": 15,
                "trainer.enable_model_summary": False,
                "trainer.num_sanity_val_steps": 0,
            }
        )


def cli_main():
    MyLightningCLI(
        BEATsTransferLearningModel, BirdDataModule, seed_everything_default=42
    )


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()