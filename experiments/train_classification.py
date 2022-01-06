import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
import pandas as pd
import random

from tqdm import tqdm
import argparse
import typing

from gradinit import GradInitWrapper


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad_init", type=bool, default=True)
    parser.add_argument("--no_grad_init", dest="grad_init", action="store_false")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--dataset_save_path", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--model", type=str, default="resnet152")
    parser.add_argument("--model_pretrained", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment_csv_log", type=str, default="logs/log.csv")

    args = parser.parse_args()
    return args


def fix_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_transforms() -> typing.Tuple[transforms.Compose, transforms.Compose]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    return transform_train, transform_test


def create_dataloaders(
    dataset: str, dataset_save_path: str, batch_size: int, workers: int
) -> typing.Tuple[DataLoader, DataLoader, int]:
    if dataset == "cifar10":
        vision_ds = datasets.CIFAR10
        classes = 10
    elif dataset == "cifar100":
        vision_ds = datasets.CIFAR100
        classes = 100
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    train_transform, test_transform = create_transforms()

    train_ds = vision_ds(
        root=dataset_save_path, train=True, download=True, transform=train_transform
    )
    test_ds = vision_ds(
        root=dataset_save_path, train=False, download=True, transform=test_transform
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, num_workers=workers, shuffle=True
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, num_workers=workers, shuffle=False
    )

    return train_dl, test_dl, classes


def create_model(
    model_name: str, pretrained: bool, classes_count: int
) -> torch.nn.Module:
    return models.__dict__[model_name](pretrained=pretrained, num_classes=classes_count)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_optimizer(
    args: argparse.Namespace, model: torch.nn.Module
) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01 if args.lr is None else args.lr,
            momentum=args.sgd_momentum,
        )

    if args.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=0.001 if args.lr is None else args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
        )

    raise ValueError(f"Unknown optimizer {args.optimizer}")


def train_epoch(
    model: torch.nn.Module,
    device: torch.device,
    dl: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    epoch_progress = tqdm(dl)
    it = 0
    sum_loss = 0
    for images, targets in epoch_progress:
        images = images.to(device)
        targets = targets.to(device)
        pred = model(images)
        loss = torch.nn.functional.cross_entropy(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        it += 1
        epoch_progress.set_description(f"Iteration {it}. Loss={sum_loss / it : .5f}")

    return sum_loss / it


def validate_epoch(
    model: torch.nn.Module, device: torch.device, dl: DataLoader
) -> typing.Tuple[float, float]:
    model.eval()
    epoch_progress = tqdm(dl)
    it = 0
    sum_loss = 0
    correct = 0
    total = 0
    for images, targets in epoch_progress:
        images = images.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            pred = model(images)
            loss = torch.nn.functional.cross_entropy(pred, targets)
            correct += (torch.argmax(pred, dim=1) == targets).sum().item()

        sum_loss += loss.item()
        it += 1
        total += targets.shape[0]
        epoch_progress.set_description(
            f"Iteration {it}. val_loss={sum_loss / it : .5f}, val_acc={float(correct) / total: .3f}"
        )

    return sum_loss / it, float(correct) / total


def train_model(
    model: torch.nn.Module,
    device: torch.device,
    train_dl: DataLoader,
    val_dl: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    logs: typing.Optional[typing.Dict[str, typing.List[float]]] = None,
) -> typing.Dict[str, typing.List[float]]:
    if logs is None:
        logs = {"loss": [], "val_loss": [], "val_acc": [], "epoch": []}
        epoch_shift = 0
    else:
        epoch_shift = logs["epoch"][-1] + 1

    epochs_progress = tqdm(range(epochs), desc="Epoch 0")
    for epoch in epochs_progress:
        train_loss = train_epoch(model, device, train_dl, optimizer)
        val_loss, val_acc = validate_epoch(model, device, val_dl)

        epochs_progress.set_description(
            f"Epoch {epoch}. Loss={train_loss : .5f}. Val_loss={val_loss: .5f}. Val_acc={val_acc: .3f}"
        )
        logs["loss"].append(train_loss)
        logs["val_loss"].append(val_loss)
        logs["val_acc"].append(val_acc)
        logs["epoch"].append(epoch + epoch_shift)

    return logs


def grad_init(
    model: torch.nn.Module,
    device: torch.device,
    dl: DataLoader,
    args: argparse.Namespace,
) -> float:
    # Wrap model to the gradinit wrapper
    with GradInitWrapper(model) as ginit:
        optimizer = get_optimizer(args, model)
        norm = 2 if args.optimizer == "sgd" else 1

        model.train()
        epoch_progress = tqdm(dl)
        it = 0
        sum_loss = 0
        for images, targets in epoch_progress:
            images = images.to(device)
            targets = targets.to(device)
            pred = model(images)
            loss = torch.nn.functional.cross_entropy(pred, targets)
            sum_loss += loss.item()

            # recalculate gradinit loss based on your loss
            loss = ginit.grad_loss(loss, norm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # clip minimum values for the init scales
            ginit.clamp_scales()

            it += 1
            epoch_progress.set_description(
                f"Gradinit. Iteration {it}. Loss={sum_loss / it : .5f}"
            )

        return sum_loss / it


def save_logs(logs: typing.Dict[str, typing.List[float]], path: str):
    pd.DataFrame(logs).to_csv(path, index=False)


def run_experiment(args: argparse.Namespace):
    fix_seed(args.seed)
    train_dl, test_dl, classes_count = create_dataloaders(
        args.dataset, args.dataset_save_path, args.batch_size, args.workers
    )
    model = create_model(args.model, args.model_pretrained, classes_count)
    device = get_device()
    model.to(device)

    logs = None
    if args.grad_init:
        ginit_loss = grad_init(model, device, train_dl, args)
        val_loss, val_acc = validate_epoch(model, device, test_dl)
        logs = {
            "loss": [ginit_loss],
            "val_loss": [val_loss],
            "val_acc": [val_acc],
            "epoch": [0],
        }

    train_optim = get_optimizer(args, model)
    logs = train_model(
        model, device, train_dl, test_dl, train_optim, args.epochs, logs=logs
    )

    save_logs(logs, args.experiment_csv_log)


if __name__ == "__main__":
    args = arguments()

    run_experiment(args)
