import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import LFW_DATASET, train_test_split
from src.models.convnet import ConvNet

from loguru import logger


class TrainTestLoop:

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        epochs,
        loss_fn,
        optimizer,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_loop(self):
        self.model.train()
        running_loss = 0
        steps = 0
        loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)

        for x1, x2, y in loader:
            # forwarding das imagens
            preds = self.model.forward(x1, x2)

            # calcula loss
            loss = self.loss_fn(preds.squeeze(), y)
            running_loss += loss.item()

            # retropropaga o erro
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            steps += 1

        return running_loss/steps

    def test_loop(self):
        running_loss = 0
        steps = 0

        self.optimizer.zero_grad()

        with torch.no_grad():
            self.model.eval()

            loader = DataLoader(self.val_dataset, batch_size=len(self.val_dataset))

            for x1, x2, y in loader:
                preds = self.model(x1, x2)
                loss = self.loss_fn(preds.squeeze(), y)
                running_loss += loss.item()
                steps += 1

        return running_loss/steps

    def fit(self):
        logger.info("Starting model training")

        train_loss = []
        val_loss = []

        for i in range(self.epochs):
            _tloss = self.train_loop()
            _vloss = self.test_loop()
            train_loss.append(_tloss)
            val_loss.append(_vloss)

            logger.info(f"Epoch: {i} - Train loss: {_tloss} - Val loss: {_vloss}")

        return train_loss, val_loss


# instancia modelo e conjunto de dados
model = ConvNet()

dataset = LFW_DATASET
dataset.load_data()

model.to("cuda")
dataset.to_cuda()

# separa em treino e teste
train_dataset, test_dataset = train_test_split(dataset, test_size=.2)

loss_fn  = nn.BCELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=3e-4, momentum=.5)

train_loop = TrainTestLoop(
    model=model,
    train_dataset=train_dataset,
    val_dataset=test_dataset,
    epochs=10000,
    loss_fn=loss_fn,
    optimizer=optimizer
)

train_loop.fit()