import sys
import time
import warnings
from pathlib import Path

import torch

from reasoning_from_scratch.qwen3 import (
    QWEN_CONFIG_06_B,
    KVCache,
    Qwen3Model,
    Qwen3Tokenizer,
    download_qwen3_small,
)

MODEL_DIR = Path("qwen3")


def get_device(enable_tensor_cores: bool = True) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")
        if enable_tensor_cores:
            major, minor = map(int, torch.__version__.split(".")[:2])
            if (major, minor) >= (2, 9):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
        print("Using Intel GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


@torch.inference_mode()
def generate_text_basic_stream_cache(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

        yield next_token

        out = model(next_token, cache=cache)[:, -1]

def generate_text_stream_concat(model, tokenizer, prompt, device, max_new_tokens, verbose=False):
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    generate_ids = []
    for token in generate_text_basic_stream_cache(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id
    ):
        next_token_id = token.squeeze(0)
        generate_ids.append(next_token_id.item())

        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )

    return tokenizer.decode(generate_ids)


def generate_stats(output_token_ids, tokenizer, start_time, end_time):
    total_time = end_time - start_time
    print(f"\n\nTime: {total_time:.2f} sec")
    print(f"{int(output_token_ids.numel() / total_time)} tokens/sec")
    for name, backend in (("CUDA", getattr(torch, "cuda", None)), ("XPU", getattr(torch, "xpu", None))):
        if backend is None or not backend.is_available():
            continue
        device_type = output_token_ids.device.type
        if device_type != name.lower():
            warnings.warn(
                f"{name} is available but tensors are on {device_type}. Memory stats may be 0."
            )
        if hasattr(backend, "synchronize"):
            backend.synchronize()
        max_mem_gb = backend.max_memory_allocated() / (1024 ** 3)
        print(f"Max {name} memory allocated: {max_mem_gb:.2f} GB")
        backend.reset_peak_memory_stats()


def compile_model(model):
    """Wrap torch.compile with Python version guard (Dynamo requires < 3.12)."""
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 8):
        # Avoids retriggering recompilations when model has self.pos = self.pos + 1
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
    if sys.version_info >= (3, 12):
        return model  # torch.compile not supported on Python 3.12+
    return torch.compile(model)


def load_model(device: torch.device) -> Qwen3Model:
    model_path = MODEL_DIR / "qwen3-0.6B-base.pth"
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model


def load_tokenizer() -> Qwen3Tokenizer:
    tokenizer_path = MODEL_DIR / "tokenizer-base.json"
    return Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)


def load_model_and_tokenizer(which_model, device,  use_compile, local_dir="qwen3"):
    if which_model == "base":
        tokenizer_path = Path(local_dir) / "tokenizer-base.json"
        model_path = Path(local_dir) / "qwen3-0.6B-base.pth"
        if not (tokenizer_path.exists() and model_path.exists()):
            download_qwen3_small(kind="base", tokenizer_only=False, out_dir=local_dir)
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    elif which_model == "reasoning":
        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        model_path = Path(local_dir) / "qwen3-0.6B-reasoning.pth"
        if not (tokenizer_path.exists() and model_path.exists()):
            download_qwen3_small(kind="reasoning", tokenizer_only=False, out_dir=local_dir)
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path, apply_chat_template=True, add_generation_prompt=True, add_thinking=True)
    else:
        raise ValueError(f"Invalid choice: which_model={which_model}")
    
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    if use_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    return model, tokenizer    

if __name__ == "__main__":
    # download_qwen3_small(kind="base", tokenizer_only=False, out_dir=str(MODEL_DIR))

    device = torch.device("cpu")  # or: device = get_device()
    tokenizer = load_tokenizer()
    model = load_model(device)
    model_compiled = compile_model(model)

    prompt = "Explain large language models in a single sentence."
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    max_new_tokens = 100

    start_time = time.time()
    generated_ids = []
    for token in generate_text_basic_stream_cache(
        model=model_compiled,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        token_id = token.squeeze(0).tolist()
        print(tokenizer.decode(token_id), end="", flush=True)
        generated_ids.append(token.squeeze(0))

    end_time = time.time()
    output_token_ids = torch.cat(generated_ids, dim=0)
    generate_stats(output_token_ids, tokenizer, start_time, end_time)
