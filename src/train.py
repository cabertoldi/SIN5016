import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import LFW_DATASET, train_test_split
from src.models.convnet import ConvNet

import pandas as pd

from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import KFold

from loguru import logger
from itertools import product

writer = SummaryWriter("runs/large-v5")


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

    def evaluate_early_stoping(
        self, current_val_loss: np.float64, current_acc: np.float64
    ):
        if current_val_loss < self.lower_loss:
            self.best_params = self.model.state_dict()
            self.epochs_without_improvement = 0
            self.lower_loss = current_val_loss
            self.best_acc = current_acc
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

        self.evaluate_early_stoping(running_loss / steps, running_acc / steps)

        return running_loss / steps, running_acc / steps

    def fit(self):
        logger.info("Starting model training")

        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []

        for i in range(self.epochs):
            _tloss, _tacc = self.train_loop()
            _vloss, _vacc = self.test_loop()
            train_loss.append(_tloss)
            val_loss.append(_vloss)
            train_acc.append(_tacc)
            val_acc.append(_vacc)

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

        return train_loss, val_loss, train_acc, val_acc


class CrossValidation:
    def __init__(self, model, dataset, epochs, cv, parameters):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.cv = cv
        self.parameters = parameters

    def cross_validate(self):

        accuracies = []
        kfold = KFold(n_splits=self.cv)

        for train_idx, val_idx in kfold.split(self.dataset):

            train_dataset = Subset(dataset=self.dataset, indices=train_idx)
            val_dataset = Subset(dataset=self.dataset, indices=val_idx)

            model = self.model(**self.parameters)

            if torch.cuda.is_available():
                model.to("cuda")

            loss_fn = nn.BCELoss()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-3)

            tloop = TrainTestLoop(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=self.epochs,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )

            tloop.fit()
            accuracies.append(tloop.best_acc)

        return accuracies


def run_cross_val(dataset, cv):

    conv_dropout = [0.1, 0.2, 0.3, 0.5, 0.75]
    dense_dropout = [0.1, 0.2, 0.3, 0.5, 0.75]
    parameters = list(product(conv_dropout, dense_dropout))
    best_acc = -np.inf

    p1 = []
    p2 = []
    acc = []
    std = []

    for conv_drop, dense_drop in parameters:
        logger.info(
            f"Running for params: conv_drop = {conv_drop} - dense_drop = {dense_drop}"
        )

        crossval = CrossValidation(
            model=ConvNet,
            dataset=dataset,
            epochs=10_000,
            parameters={
                "conv_dropout": conv_drop,
                "dense_dropout": dense_drop,
            },
            cv=cv,
        )

        results = crossval.cross_validate()
        accuracy = np.mean(results)

        if accuracy > best_acc:
            best_acc = accuracy
            best_params = {"conv_dropout": conv_drop, "dense_dropout": dense_drop}

        p1.append(conv_drop)
        p2.append(dense_drop)
        acc.append(np.mean(results))
        std.append(np.std(results))

    df = pd.DataFrame({"conv_dropout": p1, "dense_dropout": p2, "acc": acc, "std": std})
    df["model"] = "ConvNet"
    df.to_csv("execucao/convnet.csv")

    return best_params


def main():
    dataset = LFW_DATASET
    dataset.load_data()
    dataset.to_cuda()

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.3)
    train_dataset, validation_dataset = train_test_split(train_dataset, test_size=0.2)

    # get best params from cross validation
    logger.info("Starting cross validation pipeline")
    best_params = run_cross_val(train_dataset, cv=5)

    # full train
    logger.info("Training with the complete training set")
    model = ConvNet(**best_params)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-3)

    tloop = TrainTestLoop(
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        epochs=10_000,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )

    tloop.fit()
    logger.info(f"Accuracy on validation set: {tloop.best_acc}")

    # submit test data to model


if __name__ == "__main__":
    main()
