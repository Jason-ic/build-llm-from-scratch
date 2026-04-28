import torch
import time
from pathlib import Path
from pprint import pprint
from reasoning_from_scratch.qwen3 import KVCache
from load_model import get_device, load_model_and_tokenizer
from load_dataset import load_math_train
from evaluate_model import render_prompt, extract_final_candidate, grade_answer
from self_refine import generate_text_stream_concat_flex, generate_text_top_p_stream_cache
from self_consistency import top_p_filter


@torch.no_grad()
def sample_response(model, tokenizer, prompt, device, max_new_tokens=512, temperature=0.8, top_p=0.9):
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        device=device
    )
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(input_ids.unsqueeze(0), cache=cache)[:, -1]
    generated = []

    for _ in range(max_new_tokens):
        if temperature and temperature != 1.0:
            logits = logits / temperature
        probas = torch.softmax(logits, dim=-1)
        probas = top_p_filter(probas, top_p)
        next_token = torch.multinomial(
            probas.cpu(), num_samples=1
        ).to(device)
        token_id = next_token.item()
        generated.append(token_id)

        if (
            tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id
        ):
            break
        logits = model(next_token, cache=cache)[:, -1]

    full_token_ids = torch.cat(
        [input_ids, torch.tensor(generated, device=device, dtype=input_ids.dtype),]
    )
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)

def reward_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(
        answer_text, fallback=None 
    )
    if not extracted:
        return 0.0
    correct = grade_answer(extracted, ground_truth)
    return float(correct)

@torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)

    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    logprobs = torch.log_softmax(logits, dim=-1)
    start = len(prompt_ids) - 1
    end = full_ids.shape[0] - 1

    t_idx = torch.arange(start, end, device=device)
    next_tokens = full_ids[start + 1 : end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]

    return torch.mean(next_token_logps).item()

def sequence_logprob(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    selected = logprobs[:-1].gather(
        1, token_ids[1:].unsqueeze(-1)
    ).squeeze(-1)

    return torch.sum(selected[prompt_len - 1:])

def compute_grpo_loss(
    model, tokenizer, example, device,
    num_rollouts=2, max_new_tokens=256, 
    temperature=0.8, top_p=0.9
):
    assert num_rollouts >= 2
    roll_logps, roll_rewards, samples = [], [], []
    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()

    for _ in range(num_rollouts):
        token_ids, prompt_len, text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        reward = reward_rlvr(text, example["answer"])
        logp = sequence_logprob(model, token_ids, prompt_len)
        roll_logps.append(logp)
        roll_rewards.append(reward)
        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
            }
        )
    if was_training:
        model.train()

    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    logps = torch.stack(roll_logps)
    pg_loss = -(advantages.detach() * logps).mean()
    loss = pg_loss

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "samples": samples,
        "loss_tensor": loss
    }

def train_rlvr_grpo(model, tokenizer, math_data, device, steps=None,
                    num_rollouts=2, max_new_tokens=256, temperature=0.8,
                    top_p=0.9, lr=1e-5, checkpoint_every=50, checkpoint_dir=".", csv_log_path=None):
    if steps is None:
        steps = len(math_data)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    current_step = 0

    if csv_log_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_log_path = f"train_rlvr_grpo_metrics_{timestamp}.csv"
    csv_log_path = Path(csv_log_path)

    try:
        for step in range(steps):
            optimizer.zero_grad()
            current_step = step + 1
            example = math_data[step % len(math_data)]

            stats = compute_grpo_loss(
                model=model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            stats["loss_tensor"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            step_tokens = sum(sample["gen_len"] for sample in stats["samples"])
            avg_response_len = (step_tokens / len(stats["samples"]) if stats["samples"] else 0.0)
            append_csv_metrics(csv_log_path, current_step, steps, stats["loss"], reward_avg, avg_response_len)
            print(
                f"[Step {current_step}/{steps}] "
                f"loss={stats['loss']:.4f} "
                f"reward_avg={reward_avg:.3f} "
                f"avg_resp_len={avg_response_len:.1f}"
            )

            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = save_checkpoint(model=model, checkpoint_dir=checkpoint_dir, step=current_step)
            
    except KeyboardInterrupt:
        ckpt_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=max(1, current_step),
            suffix="interrupt",
        )
        print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt_path}")
        return model
    return model

def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = (
        checkpoint_dir /
        f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path

def append_csv_metrics(csv_log_path, step_idx, total_steps, loss, reward_avg, avg_response_len):
    if not csv_log_path.exists():
        csv_log_path.write_text(
            "step,total_steps,loss,reward_avg,avg_response_len\n",
            encoding="utf-8",
        )
    with csv_log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{step_idx},{total_steps},{loss:.6f},{reward_avg:.6f},"
            f"{avg_response_len:.6f}\n"
        )

if __name__ == "__main__":
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
    prompt = render_prompt(prompt=raw_prompt)
    print("=== PROMPT ===")
    print(prompt)
    torch.manual_seed(3)
    print("=== RESPONSE ===")
    # response = generate_text_stream_concat_flex(
    #     model, tokenizer, prompt, device, 
    #     max_new_tokens=2048, verbose=True, 
    #     generate_func=generate_text_top_p_stream_cache,
    #     temperature=0.9, top_p=0.9
    # )
    # token_ids, prompt_len, answer_text = sample_response(
    #     model=model,
    #     tokenizer=tokenizer,
    #     prompt=prompt,
    #     device=device,
    #     max_new_tokens=512,
    #     temperature=0.9,
    #     top_p=0.9,
    # )
    # print(answer_text)
    # rollouts = [
    #     r"\boxed{83}",
    #     r"The correct answer is \boxed{83}",
    #     r"The final answer is 83",
    #     r"We get \boxed{38}",
    # ]
    # rollout_rewards = []
    # for answer in rollouts:
    #     reward = reward_rlvr(answer_text=answer, ground_truth="83")
    #     print(f"Answer: {answer!r}")
    #     print(f"Reward: {reward}\n")
    #     rollout_rewards.append(reward)

    # avg_logprob_val = avg_logprob_answer(
    #     model, tokenizer,
    #     prompt=prompt,
    #     answer=answer_text,
    #     device=device
    # )
    # print("=== logprob val ===")
    # print(avg_logprob_val)
    # sequence_logprob_val = avg_logprob_val * (
    #     len(tokenizer.encode(answer_text))
    # )
    # print(sequence_logprob_val)
    # print(sequence_logprob(model, token_ids, prompt_len))
    # rollout_logps = []
    # for text in rollouts:
    #     token_ids = tokenizer.encode(prompt + " " + text)
    #     logprob = sequence_logprob(
    #         model=model,
    #         token_ids=torch.tensor(token_ids, device=device),
    #         prompt_len=prompt_len,
    #     )
    #     print(f"Answer: {text}")
    #     print(f"Logprob: {logprob.item():.4f}\n")
    #     rollout_logps.append(logprob)

    # torch.manual_seed(123)
    math_train = load_math_train()
    # stats = compute_grpo_loss(
    #     model=model,
    #     tokenizer=tokenizer,
    #     example=math_train[4],
    #     device=device,
    #     num_rollouts=2,
    #     max_new_tokens=256,
    #     temperature=0.8,
    #     top_p=0.9
    # )
    # pprint(stats)

    # device = get_device()
    # model.to(device)
    device = torch.device("cpu")
    torch.manual_seed(0)

    train_rlvr_grpo(
        model=model,
        tokenizer=tokenizer,
        math_data=math_train,
        device=device,
        steps=50,
        num_rollouts=4,
        max_new_tokens=512,
        temperature=0.8,
        top_p=0.9,
        lr=1e-5,
        checkpoint_every=5,
        checkpoint_dir=".",
    )
        