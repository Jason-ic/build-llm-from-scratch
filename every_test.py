import tiktoken
import torch
from llm_arch.gpt import DummyGPTModel
from train import generate_text_simple

# GPT_CONFIG_124M = "/Users/youfangdajiankang/build-llm-from-scratch/config.yaml"
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# tokenizer = tiktoken.get_encoding("gpt2")
# batch = []
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"
# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch = torch.stack(batch, dim=0)
# print(batch)

# torch.manual_seed(123)
# model = DummyGPTModel(GPT_CONFIG_124M)
# model.eval()
# logits = model(batch)
# print("Output shape:", logits.shape)
# print(logits)
# def text_to_token_ids(text, tokenizer):
#     encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
#     encoded_tensor = torch.tensor(encoded).unsqueeze(0)
#     return encoded_tensor

# def token_ids_to_text(token_ids, tokenizer):
#     flat = token_ids.squeeze(0)
#     return tokenizer.decode(flat.tolist())

# start_context = "Every effort moves you"
# tokenizer = tiktoken.get_encoding("gpt2")

# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids(start_context, tokenizer),
#     max_new_tokens=10,
#     context_size=GPT_CONFIG_124M["context_length"]
# )
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

import pandas as pd
df = pd.read_csv("/Users/youfangdajiankang/build-llm-from-scratch/classify-datasets/sms_spam_collection/SMSSpamCollection.tsv", sep="\t", header=None, names=["Label", "Text"])
print(df)