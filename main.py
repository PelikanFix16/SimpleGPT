import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm.auto import tqdm


from gpt import Gpt


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding="max_length",
    )


generation_configs = {
    "code_generation": {"temperature": 0.2, "top_k": 50, "top_p": 0.1},
    "creative_writing": {"temperature": 0.7, "top_k": 40, "top_p": 0.8},
    "chatbot": {"temperature": 0.5, "top_k": 40, "top_p": 0.5},
    "code_comment_generation": {"temperature": 0.3, "top_k": 30, "top_p": 0.2},
    "data_analysis_scripting": {"temperature": 0.2, "top_k": 50, "top_p": 0.1},
    "exploratory_code_writing": {"temperature": 0.6, "top_k": 40, "top_p": 0.7},
}

use_case = "creative_writing"
config = generation_configs[use_case]

test_mode = False


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):

    top_k = min(top_k, logits.size(-1))

    if top_k > 0:

        topk_values, _ = torch.topk(logits, top_k, dim=-1)
        threshold = topk_values[:, -1].unsqueeze(-1)
        indices_to_remove = logits < threshold
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        mask = torch.zeros_like(logits, dtype=torch.bool)

        mask.scatter_(1, sorted_indices, sorted_indices_to_remove)

        logits = logits.masked_fill(mask, filter_value)

    return logits


def generate_text(
    model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=0, top_p=0.0
):
    model.eval()
    model.to("cuda")

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
    seq_len = input_ids.size(1)

    for _ in range(max_new_tokens):

        logits = model(input_ids)

        next_token_logits = logits[:, -1, :] / temperature
        filtered_logits = top_k_top_p_filtering(
            next_token_logits, top_k=top_k, top_p=top_p
        )

        next_token = torch.multinomial(
            F.softmax(filtered_logits, dim=-1), num_samples=1
        )

        input_ids = torch.cat((input_ids, next_token), dim=1)
        seq_len += 1

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


def main():

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("bookcorpus", split="train[:20%]")

    def hf_tokenize_fn(examples):
        return tokenize_function(examples, tokenizer)

    tokenized_dataset = dataset.map(
        hf_tokenize_fn,
        batched=True,
        remove_columns=["text"],
        num_proc=24,
    )

    tokenized_dataset.set_format(type="torch", columns=["input_ids"])

    train_loader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

    vocab_size = len(tokenizer)
    d_model = 768
    ff_layers = 1024
    num_heads = 8
    seq_len = 1024
    dropout = 0.2
    num_layers = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()

    model = Gpt(
        vocab_size=vocab_size,
        d_model=d_model,
        ff_layers=ff_layers,
        num_heads=num_heads,
        seq_len=seq_len,
        dropout=dropout,
        num_layers=num_layers,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    num_epochs = 2
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    checkpoint_path = "last_checkpoint.pth"
    start_epoch = 0
    global_step = 0

    if os.path.exists(checkpoint_path):
        print(f"[INFO] Wznawianie z {checkpoint_path} ...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        print(f"[INFO] Wznowiono od epoch={start_epoch}, global_step={global_step}")

    # 9. Trening
    if not test_mode:
        model.train()
        save_every = 1000

        for epoch in range(start_epoch, num_epochs):
            print(f"=== EPOCH {epoch+1}/{num_epochs} ===")
            epoch_loss = 0.0

            pbar = tqdm(train_loader, total=len(train_loader))
            for batch_idx, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(device)

                if input_ids.size(1) < 2:
                    continue
                inputs = input_ids[:, :-1]
                labels = input_ids[:, 1:]

                logits = model(inputs)

                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), labels.reshape(-1)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                pbar.set_description(f"Loss: {loss.item():.4f}")

                if global_step % save_every == 0:
                    save_dict = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }
                    torch.save(save_dict, checkpoint_path)
                    print(f"[INFO] Zapisano checkpoint przy kroku={global_step}")

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} done. Avg loss={avg_loss:.4f}")
    else:
        print("Generating text samples...")
        sample_prompt = "She was walking down the street"
        generated = generate_text(
            model,
            tokenizer,
            sample_prompt,
            max_new_tokens=20,
            temperature=config["temperature"],
            top_k=config["top_k"],
            top_p=config["top_p"],
        )
        print(f"[GEN-EPOCH] Prompt: {sample_prompt}")
        print(f"            Generated: {generated}\n")

    print("Koniec treningu!")


if __name__ == "__main__":
    main()
