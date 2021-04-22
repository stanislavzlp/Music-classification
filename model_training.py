"""
Основной цикл обучения модели
"""

from functools import partial

import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.nn import BCELoss
from torch.optim import Adam
from tqdm import tqdm

from data_loaders import train_dataloader, val_dataloader
from model import resnet_model
from tools import TrainEpoch, ValidEpoch

EPOCHS = 15

loss = BCELoss()  # функция потерь

# Определим метрики, за которыми будем следить во время обучения:
f1_multiclass = partial(f1_score, average="samples")  # f1-метрика
f1_multiclass.__name__ = 'f1'
accuracy = accuracy_score  # метрика accuracy
accuracy.__name__ = 'accuracy'


optimizer = Adam(resnet_model.parameters())  # Оптимизатор
device = 'cuda:0'

train_epoch = TrainEpoch(
        resnet_model,
        loss=loss,
        metrics=[f1_multiclass, accuracy_score],
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

valid_epoch = ValidEpoch(
    resnet_model,
    loss=loss,
    metrics=[f1_multiclass, accuracy_score],
    device=device,
    verbose=True,
)

for i in tqdm(range(EPOCHS)):
    print(f'\nEpoch: {i + 1}')

    train_logs = train_epoch.run(train_dataloader)
    valid_logs = valid_epoch.run(val_dataloader)


torch.save(resnet_model, f'saved_model_{valid_logs["f1"]}')  # Пишем в название модели f1 score на валидационной выборке


