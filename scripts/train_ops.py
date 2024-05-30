# -*- coding: utf=8 -*-

import torch
from torch import nn

from sklearn.metrics import accuracy_score

__all__ = [
    'training_loop_with_batch'
]


def training_loop_with_batch(
    model,
    dataset,
    optimizer,
    scheduler,
    epochs,
    criterion = nn.BCEWithLogitsLoss(),
    device = "cpu"
):
    TRAIN_LOSSES, TEST_LOSSES = [], []
    TRAIN_ACCS, TEST_ACCS = [], []

    for epoch in range(1, epochs + 1):
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []

        model.train()  # Set up training mode

        for train_batch in iter(dataset):
            train_batch_sequences, train_batch_targets = train_batch

            train_batch_sequences = train_batch_sequences.to(device)
            train_batch_targets = train_batch_targets.to(device)

            # y_pred = model(train_data_batch, train_mask_batch.bool())
            # y_pred = model(train_data_batch)
            with torch.cuda.amp.autocast():
                y_pred = model(train_batch_sequences)

            # Compare actual targets and predicted targets to get the loss
            train_loss = criterion(y_pred, train_batch_targets)
            train_loss.backward()  # Backpropagate the loss

            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(train_loss.item())

            train_acc = accuracy_score(train_batch_targets.cpu().numpy(), y_pred.max(dim=-1).indices.cpu().numpy())
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

    return {
        "loss": TRAIN_LOSSES,
        "accuracy": TRAIN_ACCS,
        "model": model
    }