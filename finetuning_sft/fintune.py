import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from torch.utils.data import DataLoader

from process_script.gpt_download import download_and_load_gpt2
from llm_arch.gpt import DummyGPTModel
from load_pretrain import load_weights_into_gpt

from pre_train import calc_loss_loader, train_model_simple
from dataset import train_loader, val_loader, device, val_data, tokenizer
from data_process import format_input

BASE_CONFIG = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate": 0.0, # Dropout rate
    "qkv_bias": True # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="/Users/youfangdajiankang/build-llm-from-scratch/gpt2"
)
model = DummyGPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
model.to(device)

start_time = time.time()
torch.manual_seed(123)

# with torch.no_grad():
#     train_loss = calc_loss_loader(
#         train_loader, model, device, num_batches=5
#     )
#     val_loss = calc_loss_loader(
#         val_loader, model, device, num_batches=5
#     )

# print("Training loss:", train_loss)
# print("Validation loss:", val_loss)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
num_epochs = 2

train_losses,  val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq=5,
                                                            eval_iter=5, start_context=format_input(val_data[0]), tokenizer=tokenizer)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")


