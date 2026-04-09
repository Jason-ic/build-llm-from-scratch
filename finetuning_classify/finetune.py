import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import tiktoken
import matplotlib.pyplot as plt
from dataset import SpamDataset
from torch.utils.data import DataLoader
from process_script.gpt_download import download_and_load_gpt2
from llm_arch.gpt import DummyGPTModel
from load_pretrain import load_weights_into_gpt
from train import generate_text_simple
from train import text_to_token_ids, token_ids_to_text

tokenizer = tiktoken.get_encoding("gpt2")
# print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

train_dataset = SpamDataset(csv_file="/Users/youfangdajiankang/build-llm-from-scratch/finetuning/train.csv", tokenizer=tokenizer)

# print(train_dataset.max_length)

val_dataset = SpamDataset(csv_file="/Users/youfangdajiankang/build-llm-from-scratch/finetuning/validation.csv", tokenizer=tokenizer, max_length=train_dataset.max_length)
test_dataset = SpamDataset(csv_file="/Users/youfangdajiankang/build-llm-from-scratch/finetuning/test.csv", tokenizer=tokenizer, max_length=train_dataset.max_length)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)

# for input_batch, target_batch in train_loader:
#     pass

# print(input_batch.shape)
# print(target_batch.shape)
# print(f"{len(train_loader)} training batches")
# print(f"{len(val_loader)} validation batches")
# print(f"{len(test_loader)} test batches")

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
 "vocab_size": 50257,
 "context_length": 1024,
 "drop_rate": 0.0,
 "qkv_bias": True
}
model_configs = {
 "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
 "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
 "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
 "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="/Users/youfangdajiankang/build-llm-from-scratch/gpt2")

model = DummyGPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

# text_1 = "Every effort moves you"
# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(text_1, tokenizer),
#     max_new_tokens=15,
#     context_size=BASE_CONFIG["context_length"]
# )
# print(token_ids_to_text(token_ids, tokenizer))

# text_2 = (
#  "Is the following text 'spam'? Answer with 'yes' or 'no':"
#  " 'You are a winner you have been specially"
#  " selected to receive $1000 cash or a $2000 award.'"
# )
# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(text_2, tokenizer),
#     max_new_tokens=23,
#     context_size=BASE_CONFIG["context_length"]
# )
# print(token_ids_to_text(token_ids, tokenizer))

# print(model)
for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

def calc_accuracy_loader(data_loader, model,  device,   num_batches=None):
    model.eval()
    correct_prediction, num_example = 0,  0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i,  (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predict_labels = torch.argmax(logits, dim=-1)
            num_example += predict_labels.shape[0]
            correct_prediction += (
                (predict_labels == target_batch).sum().item()
            )
        else:
            break

    return correct_prediction / num_example

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# torch.manual_seed(123)
# train_accuracy = calc_accuracy_loader(
#     train_loader, model, device, num_batches=10
# )
# val_accuracy = calc_accuracy_loader(
#     val_loader, model, device, num_batches=10
# )
# test_accuracy = calc_accuracy_loader(
#     test_loader, model, device, num_batches=10
# )
# print(f"Training accuracy: {train_accuracy*100:.2f}%")
# print(f"Validation accuracy: {val_accuracy*100:.2f}%")
# print(f"Test accuracy: {test_accuracy*100:.2f}%")

def calc_loss_batch(input_batch, target_batch, model,  device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:,  -1,  :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)

    return loss

def calc_loss_loader(data_loader, model, device, num_batch=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batch is None:
        num_batch = len(data_loader)
    else:
        num_batch = min(num_batch,  len(data_loader))

    for i,  (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batch:
            loss = calc_loss_batch(input_batch, target_batch, model,  device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batch

# with torch.no_grad():
#     train_loss = calc_loss_loader(
#         train_loader, model, device, num_batch=5
#     )
#     val_loss = calc_loss_loader(val_loader, model, device, num_batch=5)
#     test_loss = calc_loss_loader(test_loader, model, device, num_batch=5)
# print(f"Training loss: {train_loss:.3f}")
# print(f"Validation loss: {val_loss:.3f}")
# print(f"Test loss: {test_loss:.3f}")

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "f"Train loss {train_loss:.3f}, " f"Val loss {val_loss:.3f}")
                
        
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batch=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batch=eval_iter)
    model.train()
    return train_loss, val_loss
 

start_time  = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 1

train_losses, val_losses, train_accs, val_accs, example_seen = train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=50, eval_iter=5)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

def plot_values(
    epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(
        epochs_seen, val_values, linestyle="-.",
        label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, example_seen, len(train_losses))
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

