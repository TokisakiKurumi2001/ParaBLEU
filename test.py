from ParaBLEU import ParaBLEUPDataLoader, ParaBLEUPretrainedModel
dataloader = ParaBLEUPDataLoader('xlm-roberta-base', 'facebook/m2m100_418M', 128)
[train_dataloader] = dataloader.get_dataloader(batch_size=2, types=['train'])
for batch in train_dataloader:
    # print(batch)
    break
model = ParaBLEUPretrainedModel('xlm-roberta-base', 'facebook/m2m100_418M')
m, c, g = model(batch)
# print(m.shape)
# print(c.shape)
print(g.shape)