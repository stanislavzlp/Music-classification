"""
Здесь определяется основная модель для multilabel классификации:
Берём стандартный resnet, модифицируем его, чтобы он работал с одноканальными изображениями и изменяем выходной слой
с учетом количества классов (музыкальных жанров) в исходном датасете.
"""

from torchvision.models import resnet18
from torch.nn import Sequential, Sigmoid, Linear, Conv2d, Dropout

grey_resnet = resnet18()

# Изменим первый слой, чтобы на вход принималась одноканальная картинка
grey_resnet.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

resnet_model = Sequential(grey_resnet,
                          Dropout(0.5),
                          Linear(in_features=1000, out_features=170),  # всего 170 классов
                          Sigmoid()  # классы независимы друг от друга, т.к. у нас multilabel. Поэтому не softmax!
                          )
