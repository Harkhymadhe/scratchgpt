# -*- coding: utf=8 -*-

import datetime
import os

import torch
from torch import nn

from sklearn.metrics import accuracy_score

__all__ = ["training_loop_with_batch"]


def training_loop_with_batch(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epochs: int,
    criterion: nn.Module = nn.CrossEntropyLoss(),
    device: str = "cpu",
    use_last: bool = True,
    gradient_accumulation_steps: int = 1,
):
    TRAIN_LOSSES, TRAIN_ACCS, TRAIN_PERPLEXITY = [], [], []
    TEST_LOSSES, TEST_ACCS, TEST_PERPLEXITY = [], [], []

    num_steps = len(iter(train_dataloader))

    for epoch in range(1, epochs + 1):
        train_losses, train_accs, train_perplexity = [], [], []
        test_losses, test_accs, test_perplexity = [], [], []

        model.train()  # Set up training mode

        for i, train_batch in enumerate(iter(train_dataloader)):
            step = ((epoch - 1) * num_steps) + (i + 1)
            train_batch_sequences, train_batch_targets = train_batch

            train_batch_sequences = train_batch_sequences.to(device)
            train_batch_targets = train_batch_targets.to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                y_pred = model(train_batch_sequences)

                # Compare actual targets and predicted targets to get the loss
                if use_last:
                    train_loss = criterion(y_pred[:, -1, :], train_batch_targets[:, -1])
                else:
                    y_pred = y_pred.flatten(0, 1)
                    train_batch_targets = train_batch_targets.flatten()

                    train_loss = criterion(y_pred, train_batch_targets)
                    train_loss = train_loss / gradient_accumulation_steps  # Scale loss for gradient accumulation

            train_loss.backward()  # Backpropagate the loss

            if (step + 1) % gradient_accumulation_steps == 0:
                # Clip the gradient norms to 1.0
                nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=1.0, foreach=True
                )

                # Take a step with the optimizer
                optimizer.step()  # Update weights after accumulating gradients
                optimizer.zero_grad()  # Clear accumulated gradients

            train_losses.append(train_loss.item())
            train_perplexity.append(torch.exp(train_loss).item())

            # Compare actual targets and predicted targets to get the accuracy
            if use_last:
                train_acc = accuracy_score(
                    y_pred[:, -1, :].detach().cpu().argmax(dim=-1).numpy(),
                    train_batch_targets[:, -1].detach().cpu().numpy(),
                )
            else:
                train_acc = accuracy_score(
                    y_pred.detach().cpu().argmax(dim=-1).numpy(),
                    train_batch_targets.detach().cpu().numpy(),
                )
            train_accs.append(train_acc)

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = sum(train_losses) / len(train_losses)

        avg_train_acc = sum(train_accs) / len(train_accs)

        avg_train_perplexity = torch.exp(torch.tensor([avg_train_loss])).item()

        model.eval()

        with torch.no_grad():
            for test_batch in iter(test_dataloader):
                test_batch_sequences, test_batch_targets = test_batch

                test_batch_sequences = test_batch_sequences.to(device)
                test_batch_targets = test_batch_targets.to(device)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    y_pred_ = model(test_batch_sequences)

                    # Compare actual targets and predicted targets to get the loss
                    if use_last:
                        test_loss = criterion(
                            y_pred_[:, -1, :], test_batch_targets[:, -1]
                        )
                    else:
                        y_pred_ = y_pred_.flatten(0, 1)
                        test_batch_targets = test_batch_targets.flatten()

                        test_loss = criterion(y_pred_, test_batch_targets).detach()
                        test_loss = test_loss / gradient_accumulation_steps

                test_losses.append(test_loss.item())
                test_perplexity.append(torch.exp(test_loss).item())

                # Compare actual targets and predicted targets to get the accuracy
                if use_last:
                    test_acc = accuracy_score(
                        y_pred_[:, -1, :].detach().cpu().argmax(dim=-1).numpy(),
                        test_batch_targets[:, -1].detach().cpu().numpy(),
                    )
                else:
                    test_acc = accuracy_score(
                        y_pred_.detach().cpu().argmax(dim=-1).numpy(),
                        test_batch_targets.detach().cpu().numpy(),
                    )
                test_accs.append(test_acc)

            avg_test_loss = sum(test_losses) / len(test_losses)

            avg_test_acc = sum(test_accs) / len(test_accs)

            avg_test_perplexity = torch.exp(torch.tensor([avg_test_loss])).item()

        print(
            f"Epoch: {epoch} | Step: {step} | Train loss: {avg_train_loss: .3f} | Test loss: {avg_test_loss: .3f} | "
            f"Train perplexity: {avg_train_perplexity: .3f} | Test perplexity: {avg_test_perplexity: .3f} | ",
            f"Train accuracy: {100 * avg_train_acc: .3f} % | Test accuracy: {100 * avg_test_acc: .3f} % |\n",
        )

        TRAIN_LOSSES.append(avg_train_loss)
        TRAIN_ACCS.append(avg_train_acc)
        TRAIN_PERPLEXITY.append(avg_train_perplexity)

        TEST_LOSSES.append(avg_test_loss)
        TEST_ACCS.append(avg_test_acc)
        TEST_PERPLEXITY.append(avg_test_perplexity)

    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.clear_autocast_cache()

    # Get timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Persist state for model and optimizer
    folder = f"../artefacts/models/{timestamp.split('-')[0]}/"

    if not os.path.isdir(folder):
        os.mkdir(folder)

    subfolders = os.listdir(folder)

    if len(subfolders) == 0:
        subfolder = os.path.join(folder, str(len(subfolders) + 1))
    else:
        max_number = max([int(i) for i in subfolders])
        subfolder = os.path.join(folder, str(max_number + 1))

    os.mkdir(subfolder)

    print("Artefacts are located in:", subfolder)

    state = {
        "loss": [TRAIN_LOSSES, TEST_LOSSES],
        "accuracy": [TRAIN_ACCS, TEST_ACCS],
        "perplexity": [TRAIN_PERPLEXITY, TEST_PERPLEXITY],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "final_epoch_index": epochs - 1,
        "model_config_path": os.path.join(subfolder, f"model_config_{timestamp}.json"),
        "hyperparameter_config_path": os.path.join(
            subfolder, f"hyperparameter_config_{timestamp}.json"
        ),
    }

    torch.save(
        state,
        os.path.join(subfolder, f"{model.__class__.__name__}-{timestamp}.pt"),
    )

    # Return persisted results

    return {
        "loss": TRAIN_LOSSES,
        "accuracy": TRAIN_ACCS,
        "perplexity": TRAIN_PERPLEXITY,
        "model": model,
        "model_config_path": os.path.join(subfolder, f"model_config_{timestamp}.json"),
        "hyperparameter_config_path": os.path.join(
            subfolder, f"hyperparameter_config_{timestamp}.json"
        ),
    }
