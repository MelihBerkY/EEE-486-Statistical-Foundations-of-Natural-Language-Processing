import torch
import time
import datetime

import random
from tqdm import tqdm
from random import seed

import numpy as np
from sklearn.metrics import matthews_corrcoef
import torch.utils.data as Data

from datasets import load_dataset
from transformers import AutoTokenizer


def get_device():
    if torch.backends.cuda.is_built():
        print("CUDA")
        device = torch.device("cuda")
    elif torch.backends.mps.is_built():
        print("mps")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        raise Exception("GPU is not avalaible!")
    return device


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_eval_loop(
    model, loader, optimizer, scheduler, device, n_epochs=2, seed_val=42
):
    # Set the seed value all over the place to make this reproducible.
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    train_loss_values = []
    train_mcc_values = []
    val_loss_values = []
    val_mcc_values = []

    t00 = time.time()
    for epoch_i in range(0, n_epochs):
        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, n_epochs))
        print("Training...")

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in tqdm(enumerate(loader["train"])):
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()

            loss, logits = model(
                b_input_ids, attention_mask=b_input_mask, labels=b_labels
            ).loss, model(b_input_ids, attention_mask=b_input_mask).logits
            total_train_loss += loss.item()

            # Calculate the MCC for the current batch
            logits = logits.detach().cpu().numpy()
            logits = np.argmax(logits, axis=1).flatten()
            label_ids = b_labels.to("cpu").numpy()
            train_mcc = matthews_corrcoef(logits, label_ids)
            train_mcc_values.append(train_mcc)

            loss.backward()

            # Clip the norm of the gradients to 1.0, this is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(loader["train"])
        train_loss_values.append(avg_train_loss)

        print("\nAverage training loss: {0:.2f}".format(avg_train_loss))
        print("  Training MCC: {0:.2f}".format(np.mean(train_mcc_values)))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        print("\nRunning Validation...")
        t0 = time.time()
        model.eval()
        total_val_loss, val_mcc, nb_eval_steps = 0, 0, 0

        for batch in tqdm(loader["validation"]):
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                loss, logits = model(
                    b_input_ids, attention_mask=b_input_mask, labels=b_labels
                ).loss, model(b_input_ids, attention_mask=b_input_mask).logits

            total_val_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            logits = np.argmax(logits, axis=1).flatten()
            label_ids = b_labels.to("cpu").numpy()

            val_mcc += matthews_corrcoef(logits, label_ids)
            nb_eval_steps += 1

        avg_val_loss = total_val_loss / len(loader["validation"])
        val_loss_values.append(avg_val_loss)

        val_mcc = 100 * (val_mcc / nb_eval_steps)
        val_mcc_values.append(val_mcc)
        print("  Validation loss: {0:.2f}".format(avg_val_loss))
        print(" Validation MCC: {0:.2f}".format(val_mcc))
        print(" Validation took: {:}".format(format_time(time.time() - t0)))
    return train_loss_values, train_mcc_values, val_loss_values, val_mcc_values

def init_loader(max_length=16, batch_size=32, test_size=0.2, random_state=2023):
    model_checkpoint = "bert-base-uncased"

    dataset = load_dataset("glue", "cola")

    df_s = {}
    x = {}
    y = {}
    input_ids, attention_mask = {}, {}
    datasets, loader = {}, {}

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    for split in ["train", "validation", "test"]:
        df_s[split] = dataset[split].to_pandas()
        x[split] = dataset[split].to_pandas().sentence.values
        y[split] = dataset[split].to_pandas().label.values

        input = tokenizer(
            list(x[split]),
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        input_ids[split], attention_mask[split] = input.input_ids, input.attention_mask

        datasets[split] = Data.TensorDataset(
            input_ids[split], attention_mask[split], torch.LongTensor(y[split])
        )

        loader[split] = Data.DataLoader(
            datasets[split], batch_size=batch_size, shuffle=False
        )
    return loader, y


from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification


def init_objects(
    lr, n_epochs, max_length=16, dropout_rate = 0.1, batch_size=32, test_size=0.2, random_state=2023
):
    loader, _ = init_loader(max_length=max_length, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

    model.dropout.p = dropout_rate

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)

    total_steps = len(loader["train"]) * n_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    return model, loader, optimizer, scheduler