from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from ParaBLEU import ParaBLEUPDataLoader, LitParaBLEU

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_new_dummy")

    hyperparameter = {
        "encoder_ckpt": "xlm-roberta-base",
        "gen_ckpt": "facebook/m2m100_418M",
        "lr": 3e-4,
        "alpha": 4.0,
        "beta": 10.0
    }
    lit_parableu = LitParaBLEU(**hyperparameter)

    # dataloader
    parableu_pretrained_dataloader = ParaBLEUPDataLoader(
        encoder_ckpt=hyperparameter['encoder_ckpt'],
        gen_ckpt=hyperparameter['gen_ckpt'],
        max_length=128
    )
    [train_dataloader, test_dataloader, valid_dataloader] = parableu_pretrained_dataloader.get_dataloader(batch_size=64, types=["train", "test", "validation"])

    # train model
    trainer = pl.Trainer(max_epochs=2, devices=[0], accelerator="gpu", logger=wandb_logger, val_check_interval=3000)
    trainer.fit(model=lit_parableu, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(dataloaders=test_dataloader)

    # save model & tokenizer
    lit_parableu.export_model('parableu_model/v1')
