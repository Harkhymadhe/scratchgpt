# -*- coding: utf=8 -*-

import datetime
import os

import torch
from torch import nn

from sklearn.metrics import accuracy_score

__all__ = ["training_loop_with_batch"]


def training_loop_with_batch(
    model,
    dataset,
    optimizer,
    scheduler,
    epochs,
    criterion=nn.CrossEntropyLoss(),
    device="cpu",
    stack_metrics=False
):
    TRAIN_LOSSES, TRAIN_ACCS = [], []

    for epoch in range(1, epochs + 1):
        train_losses, train_accs = [], []

        model.train()  # Set up training mode

        for train_batch in iter(dataset):
            train_batch_sequences, train_batch_targets = train_batch

            train_batch_sequences = train_batch_sequences.to(device)
            train_batch_targets = train_batch_targets.to(device)

            with torch.cuda.amp.autocast():
                y_pred = model(train_batch_sequences)

            # Compare actual targets and predicted targets to get the loss
            if stack_metrics:
                train_loss = torch.stack(
                    [
                        criterion(y_pred[i], train_batch_targets[i])
                        for i in range(len(train_batch_targets))
                    ]
                )
                train_loss = torch.mean(train_loss)
            else:
                train_loss = criterion(y_pred[-1], train_batch_targets[-1])

            train_loss.backward()  # Backpropagate the loss

            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(train_loss.item())

            # Compare actual targets and predicted targets to get the accuracy
            if stack_metrics:
                train_acc = torch.stack(
                    [
                        torch.tensor(
                            [
                                accuracy_score(
                                    y_pred[i].detach().cpu().argmax(dim=-1).numpy(),
                                    train_batch_targets[i].detach().cpu().numpy(),
                                )
                            ]
                        )
                        for i in range(len(train_batch_targets))
                    ]
                )
                train_acc = torch.mean(train_acc).item()
            else:
                train_acc = accuracy_score(
                    y_pred[-1].detach().cpu().argmax(dim=-1).numpy(),
                    train_batch_targets[-1].detach().cpu().numpy(),
                )
            train_accs.append(train_acc)

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = sum(train_losses) / len(train_losses)

        avg_train_acc = sum(train_accs) / len(train_accs)

        print(
            f"Epoch: {epoch} | Train loss: {avg_train_loss: .3f} | Train accuracy: {100 * avg_train_acc: .3f} % |\n"
        )

        TRAIN_LOSSES.append(avg_train_loss)
        TRAIN_ACCS.append(avg_train_acc)

    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.clear_autocast_cache()

    # Get timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Persist state for model and optimizer
    folder = f"../artefacts/models/{timestamp.split('-')[0]}/"

    if not os.path.isdir(folder):
        os.mkdir(folder)

    torch.save(
        model.state_dict(),
        os.path.join(folder, f"{model.__class__.__name__}-{timestamp}.pt"),
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(folder, f"{optimizer.__class__.__name__}-{timestamp}.pt"),
    )

    # Return persisted results

    return {
        "loss": TRAIN_LOSSES,
        "accuracy": TRAIN_ACCS,
        "model": model,
        "optimizer": optimizer,
        "final_epoch_index": epochs - 1,
    }
