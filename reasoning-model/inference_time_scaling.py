import torch
from collections import Counter
from reasoning_from_scratch.qwen3 import KVCache
from load_model import get_device, load_model_and_tokenizer, generate_text_basic_stream_cache
from evaluate_model import render_prompt, extract_final_candidate

# device = get_device()
device = torch.device("cpu")

model, tokenizer = load_model_and_tokenizer(
    which_model="base",
    device=device,
    use_compile=False
)

raw_prompt = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)

prompt = render_prompt(raw_prompt)
print("\n" + prompt)

def generate_text_stream_concat_flex(model, tokenizer, prompt, device, max_new_tokens,
                                     verbose=False, generate_func=None, **generate_kwargs):
    if generate_func is None: 
        generate_func = generate_text_basic_stream_cache

    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    generated_ids = []
    for token in generate_func(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        **generate_kwargs, 
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())
        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )

    return tokenizer.decode(generated_ids)

# response = generate_text_stream_concat_flex(
#     model, tokenizer, prompt, device,
#     max_new_tokens=2048, verbose=True,
#     generate_func=generate_text_basic_stream_cache
# )

# prompt_cot = prompt + "\n\nExplain step by step."
# print("\n" + prompt_cot)

# response_cot = generate_text_stream_concat_flex(
#     model, tokenizer, prompt_cot, device,
#     max_new_tokens=2048, verbose=True,
# )

def scale_logits_by_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return logits / temperature

@torch.inference_mode()
def generate_text_temp_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=0.
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    out = model(token_ids, cache=cache)[:, -1] 
    for _ in range(max_new_tokens):
        orig_device = token_ids.device
        if temperature is None or temperature == 1.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)
        else:
            logits = scale_logits_by_temperature(out, temperature) 
            probas = torch.softmax(logits, dim=-1) 
            next_token = torch.multinomial(probas.cpu(), num_samples=1) 
            next_token = next_token.to(orig_device)
        if (eos_token_id is not None and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        out = model(next_token, cache=cache)[:, -1]


# torch.manual_seed(123)
# response = generate_text_stream_concat_flex(
#     model, tokenizer, prompt, device,
#     max_new_tokens=2048, verbose=True,
#     generate_func=generate_text_temp_stream_cache, 
#     temperature=1.1
# )

def top_p_filter(probs, top_p):
    if top_p is None or top_p >= 1.0:
        return probs

    sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)
    cumprobs = torch.cumsum(sorted_probs, dim=1)

    prefix = cumprobs - sorted_probs
    keep = prefix < top_p
    keep[:, 0] = True

    kept_sorted = torch.where( 
        keep, sorted_probs, 
        torch.zeros_like(sorted_probs) 
    )

    filtered = torch.zeros_like(probs).scatter(1, sorted_idx, kept_sorted)
    denom = torch.sum(filtered, dim=1, keepdim=True).clamp_min(1e-12)
    return filtered / denom

@torch.inference_mode()
def generate_text_top_p_stream_cache(model, token_ids, max_new_tokens, eos_token_id=None, temperature=0., top_p=None):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):
        orig_device = token_ids.device
        if temperature is None or temperature == 0.0:
            next_token = torch.argmax(out, dim=-1, keepdim=True)
        else:
            logits = scale_logits_by_temperature(out, temperature)
            probs = torch.softmax(logits, dim=-1)
            probs = top_p_filter(probs, top_p)
            next_token = torch.multinomial(probs.cpu(), num_samples=1)
            next_token = next_token.to(orig_device)

        if (eos_token_id is not None and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        out = model(next_token, cache=cache)[:, -1]

# torch.manual_seed(123)
# response = generate_text_stream_concat_flex(
#     model, tokenizer, prompt, device,
#     max_new_tokens=2048, verbose=True,
#     generate_func=generate_text_top_p_stream_cache,
#     temperature=0.5,
#     top_p=0.8,
# )

def self_consistency_vote(
    model, tokenizer, prompt, device,
    num_samples=10, temperature=0.8, top_p=0.9, max_new_tokens=2048,
    show_progress=True, show_long_answer=False, seed=None,
):
    full_answers, short_answers = [], []
    for i in range(num_samples):
        if seed is not None:
            torch.manual_seed(seed + i + 1)
        
        answer = generate_text_stream_concat_flex(
            model=model, tokenizer=tokenizer, prompt=prompt, device=device,
            max_new_tokens=max_new_tokens, verbose=show_long_answer,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature, top_p=top_p,
        )
        short = extract_final_candidate( 
            answer, fallback="number_then_full" 
        )
        full_answers.append(answer)
        short_answers.append(short)
        if show_progress:
            print(f"[Sample {i+1}/{num_samples}] → {short!r}")
    
    counts = Counter(short_answers)
    groups = {s: [] for s in counts}
    for idx, s in enumerate(short_answers):
        groups[s].append(idx)

    mc = counts.most_common()
    if not mc:
        majority_winners, final_answer = [], None
    else:
        top_freq = mc[0][1]
        majority_winners = [s for s, f in mc if f == top_freq]
        final_answer = mc[0][0] if len(majority_winners) == 1 else None

    return {
        "full_answers": full_answers,
        "short_answers": short_answers,
        "counts": dict(counts),
        "groups": groups,
        "majority_winners": majority_winners,
        "final_answer": final_answer,
    }

# results = self_consistency_vote(
#     model,
#     tokenizer,
#     prompt,
#     device=device,
#     num_samples=5,
#     temperature=0.8,
#     top_p=0.9,
#     max_new_tokens=2048,
#     seed=123,
#     show_progress=True,
# )

results = self_consistency_vote(
    model,
    tokenizer,
    prompt + "\n\nExplain step by step.",
    device=device,
    num_samples=5,
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=2048,
    seed=123,
    show_progress=True,
)