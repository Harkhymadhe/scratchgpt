# -*- coding: utf=8 -*-

import datetime
import os

import torch
from torch import nn

from sklearn.metrics import accuracy_score

from torch.optim import lr_scheduler

__all__ = ["training_loop_with_batch", "persist_artefacts", "restore_artefacts"]


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
    warmup_steps: int = 1000,
    chained_scheduler=False,
    use_scheduler=False,
    resume_training=False,
    resume_date=None,
    resume_epoch=-1,
):
    # Get timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    TRAIN_LOSSES, TRAIN_ACCS, TRAIN_PERPLEXITY = [], [], []
    TEST_LOSSES, TEST_ACCS, TEST_PERPLEXITY = [], [], []

    num_steps = len(iter(train_dataloader))
    lr = optimizer.param_groups[0]["lr"]

    if scheduler is None:
        if use_scheduler:
            if chained_scheduler:
                scheduler_1 = lr_scheduler.LinearLR(
                    optimizer=optimizer,
                    start_factor=1 / warmup_steps,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )

                scheduler_2 = lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=len(iter(train_dataloader)) * (epochs + 1),
                    eta_min=lr / 10,
                    verbose=True,
                )

                scheduler = lr_scheduler.ChainedScheduler(
                    schedulers=[scheduler_1, scheduler_2]
                )
            else:
                scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer=optimizer, T_0=1, eta_min=lr / 10, verbose=True
                )
        else:
            scheduler = None

    if resume_training:
        model, optimizer, scheduler, losses, accuracies, perplexities, epoch_ = (
            restore_artefacts(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                date=resume_date,
                epoch=resume_epoch,
            )
        )
        TRAIN_LOSSES, TEST_LOSSES = losses
        TRAIN_ACCS, TEST_ACCS = accuracies
        TRAIN_PERPLEXITY, TEST_PERPLEXITY = perplexities

        epochs += epoch_
    else:
        epoch_ = 1

    for epoch in range(epoch_, epochs + 1):
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
                    train_loss = (
                        train_loss / gradient_accumulation_steps
                    )  # Scale loss for gradient accumulation

            train_loss.backward()  # Backpropagate the loss

            if step % gradient_accumulation_steps == 0:
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
            if hasattr(scheduler, "T_mult"):
                scheduler.T_cur = step
                scheduler.T_i = (num_steps * epochs) - warmup_steps

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

        final_timestamp = bool(epoch == epochs)

        subfolder = persist_artefacts(
            losses=[TRAIN_LOSSES, TEST_LOSSES],
            accuracies=[TRAIN_ACCS, TEST_ACCS],
            perplexities=[TRAIN_PERPLEXITY, TEST_PERPLEXITY],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=(epoch + epoch_ if resume_training else epoch),
            timestamp=timestamp,
            final_timestamp=final_timestamp,
        )

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


def persist_artefacts(
    losses,
    accuracies,
    perplexities,
    model,
    optimizer,
    scheduler,
    epochs,
    timestamp,
    colab_dir=None,
    persist_colab=False,
    final_timestamp=False,
):
    # Persist state for model and optimizer
    folder = f"../artefacts/models/{timestamp.split('-')[0]}/"
    if persist_colab:
        colab_folder = os.path.join(
            colab_dir, f"artefacts/models/{timestamp.split('-')[0]}/"
        )

    if not os.path.isdir(folder):
        os.mkdir(folder)

    subfolders = os.listdir(folder)

    if final_timestamp:
        try:
            subfolders.remove("checkpoints")
        except:
            pass
        if len(subfolders) == 0:
            subfolder = os.path.join(folder, str(len(subfolders) + 1))
        else:
            max_number = max([int(i) for i in subfolders if i.isnumeric()])
            subfolder = os.path.join(folder, str(max_number + 1))

        os.mkdir(subfolder)
        STATE_PATH = os.path.join(
            subfolder, f"{model.__class__.__name__}-{timestamp}.pt"
        )
    else:
        subfolder = f"../artefacts/models/{timestamp.split('-')[0]}/checkpoints/"

        if os.path.exists(subfolder):
            pass
        else:
            os.mkdir(subfolder)

        num_subfolders = len(os.listdir(subfolder)) + 1
        subfolder = os.path.join(subfolder, str(num_subfolders))
        os.mkdir(subfolder)

        STATE_PATH = os.path.join(
            subfolder, f"{model.__class__.__name__}_{timestamp}.pt"
        )

    if persist_colab:
        COLAB_STATE_PATH = os.path.join(
            colab_folder, f"{model.__class__.__name__}_{timestamp}.pt"
        )

    print("Artefacts are located in:", subfolder)

    state = {
        "loss": losses,
        "accuracy": accuracies,
        "perplexity": perplexities,
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
        STATE_PATH,
    )

    if persist_colab:
        torch.save(
            state,
            COLAB_STATE_PATH,
        )
    return subfolder


def restore_artefacts(model, optimizer, scheduler, date, epoch=-1):
    if epoch == -1:
        epoch_folders = [
            int(i) for i in os.listdir(f"../artefacts/models/{date}/checkpoints")
        ]
        epoch = max(epoch_folders)

    checkpoint_dir = f"../artefacts/models/{date}/checkpoints/{epoch}"

    state_dict_path = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt")
    ][0]

    state_dict = torch.load(state_dict_path, map_location="cpu")

    model_state_dict = state_dict["model_state"]
    optimizer_state_dict = state_dict["optimizer_state"]
    scheduler_state_dict = state_dict["optimizer_state"]

    losses = state_dict["loss"]
    accuracies = state_dict["accuracy"]
    perplexities = state_dict["perplexity"]

    model.load_state_dict(state_dict=model_state_dict)

    optimizer.load_state_dict(optimizer_state_dict)

    # print(scheduler_state_dict)

    scheduler.load_state_dict(scheduler_state_dict)

    model.train()

    print(f">>> Backpropagating any accumulated gradients...\n")
    optimizer.step()
    optimizer.zero_grad()

    print(f">>> Resuming training from epoch {epoch}...\n")

    return model, optimizer, scheduler, losses, accuracies, perplexities, epoch
