from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from ReWord import ReWordDataLoader, LitReWord

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_reorder_word")

    hyperparameter = {
        'pretrained_ck': 'roberta-base',
        'method_for_layers': 'mean',
        'layers_use_from_last': 2,
        'lr': 3e-5
    }
    lit_reword = LitReWord(**hyperparameter)

    # dataloader
    reword_dataloader = ReWordDataLoader(pretrained_ck='roberta-base', max_length=25)
    [train_dataloader, test_dataloader] = reword_dataloader.get_dataloader(batch_size=128, types=["train", "test"])

    # train model
    trainer = pl.Trainer(max_epochs=90, devices=[0], accelerator="gpu", logger=wandb_logger, log_every_n_steps=20)
    trainer.fit(model=lit_reword, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
    trainer.test(dataloaders=test_dataloader)

    # save model & tokenizer
    lit_reword.export_model('reword_model/v1')
