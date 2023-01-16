import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import LFW_DATASET, train_test_split
from src.models.convnet import ConvNet

from torch.utils.tensorboard import SummaryWriter

from loguru import logger

writer = SummaryWriter("runs/conv-drop_k3-3-3-3_h512_drop30v2-75_semnorm")


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

        # parâmetros da parada antecipada
        self.max_epochs_without_improvement = 50
        self.epochs_without_improvement = 0
        self.best_params = None
        self.lower_loss = np.inf
        self.stop_training = False

    def train_loop(self):
        self.model.train()
        running_acc = 0
        running_loss = 0
        steps = 0
        loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True)

        for x1, x2, y in loader:
            # forwarding das imagens
            preds = self.model.forward(x1, x2)

            # calcula loss
            loss = self.loss_fn(preds.squeeze(), y)
            running_loss += loss.item()

            # acc
            running_acc += self.score(y, preds.squeeze())

            # retropropaga o erro
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            steps += 1

        return running_loss / steps, running_acc / steps

    def score(self, y_test, y_pred):
        y_test = y_test.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        return np.sum((y_test == y_pred)) / len(y_test)

    def evaluate_early_stoping(self, current_val_loss: np.float64):
        if current_val_loss < self.lower_loss:
            self.best_params = self.model.state_dict()
            self.epochs_without_improvement = 0
            self.lower_loss = current_val_loss
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.max_epochs_without_improvement:
            self.stop_training = True

    def test_loop(self):
        running_loss = 0
        running_acc = 0
        steps = 0
        loader = DataLoader(self.val_dataset, batch_size=len(self.val_dataset))
        self.model.eval()

        with torch.no_grad():
            for x1, x2, y in loader:
                preds = self.model(x1, x2)

                # loss
                loss = self.loss_fn(preds.squeeze(), y)
                loss_value = loss.item()
                running_loss += loss_value

                # acc
                running_acc += self.score(y, preds.squeeze())

                steps += 1

        self.evaluate_early_stoping(running_loss / steps)

        return running_loss / steps, running_acc / steps

    def fit(self):
        logger.info("Starting model training")

        train_loss = []
        val_loss = []

        for i in range(self.epochs):
            _tloss, _tacc = self.train_loop()
            _vloss, _vacc = self.test_loop()
            train_loss.append(_tloss)
            val_loss.append(_vloss)

            writer.add_scalar("Loss/train", _tloss, i)
            writer.add_scalar("Loss/val", _vloss, i)
            writer.flush()

            logger.info(
                f"Epoch: {i} - Train loss: {_tloss} - Train acc: {_tacc} - Val loss: {_vloss} - Val acc: {_vacc} - Lower val loss = {self.lower_loss}"
            )

            if self.stop_training:
                logger.warning(
                    "Número máximo de épocas sem melhoria atingido. Interrompendo o treinamento."
                )
                break

        return train_loss, val_loss


# instancia modelo e conjunto de dados
model = ConvNet()

dataset = LFW_DATASET
dataset.load_data()

model.to("cuda")
dataset.to_cuda()

# separa em treino e teste
train_dataset, test_dataset = train_test_split(dataset, test_size=0.3)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-3)

train_loop = TrainTestLoop(
    model=model,
    train_dataset=train_dataset,
    val_dataset=test_dataset,
    epochs=10_000,
    loss_fn=loss_fn,
    optimizer=optimizer,
)

train_loop.fit()
