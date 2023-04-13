from ParaBLEU import ParaBLEUPDataLoader
dataloader = ParaBLEUPDataLoader('xlm-roberta-base', 'facebook/m2m100_418M', 128)
[train_dataloader] = dataloader.get_dataloader(batch_size=2, types=['train'])
for batch in train_dataloader:
    print(batch)
    break
