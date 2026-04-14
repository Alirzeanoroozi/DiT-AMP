import math
import os
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from Bio import SeqIO
from tqdm import tqdm
from transformers import get_scheduler, AutoModelForCausalLM, AutoTokenizer
import wandb
import torch
import numpy as np

from d3pm_runner import D3PM
from dit import DDiT_Llama
from macrel.run_macrel import run_macrel

# Avoid tokenizers fork-parallelism warning/deadlock risk with DataLoader workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def read_fasta(path):
    records_raw = list(SeqIO.parse(path, "fasta"))
    return [str(x.seq) for x in records_raw]

class SeqDataset(Dataset):
    def __init__(self, data_path, max_length=50):
        self.data = read_fasta(data_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        macrel_amp_prob, macrel_hemo_prob = run_macrel(seq)
        
        # use byte encoding
        seq = [b - ord('A') + 1 for b in seq.encode("utf-8")]
        if len(seq) < self.max_length:
            seq += [0] * (self.max_length - len(seq))
        else:
            seq = seq[: self.max_length]
        input_ids = torch.tensor(seq, dtype=torch.long)

        return {"input_ids": input_ids, "cond_amp": macrel_amp_prob, "cond_hemo": macrel_hemo_prob}


model_id = "hugohrban/progen2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).cuda()
model.eval()

def ppl(sequences):
    def compute_per_residue_probs(seq):
        """For each position, mask and calculate the probability for the correct residue."""
        probs = []
        for i in range(len(seq)):
            # Mask position i by replacing it with a special mask token if available, else use 'X'
            tokens = list(seq)
            original_res = tokens[i]
            # Attempt to use mask token if in vocab
            if tokenizer.mask_token:
                tokens[i] = tokenizer.mask_token
            else:
                tokens[i] = "X"
            masked_seq = "".join(tokens)
            inputs = tokenizer(masked_seq, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # shape (1, seq_len, vocab_size)
            # Token index for original residue at position i in tokenized input
            if hasattr(tokenizer, "convert_tokens_to_ids"):
                orig_token_id = tokenizer.convert_tokens_to_ids(original_res)
            else:
                orig_token_id = tokenizer.vocab.get(original_res, None)
            # Find the token position in the batch that corresponds to i-th residue
            # For ProGen and similar, input length matches seq_len
            token_pos = i
            probs_for_res = torch.softmax(logits[0, token_pos], dim=-1)
            if orig_token_id is not None and 0 <= orig_token_id < probs_for_res.shape[-1]:
                prob = probs_for_res[orig_token_id].item()
            else:
                prob = 0.0  # fallback if residue not in vocab
            probs.append(prob)
        return probs

    # For all generated sequences, compute per-residue probabilities
    all_per_residue_probs = [compute_per_residue_probs(s) for s in sequences]
    # Calculate mean and std for each sequence's product probability, then aggregate
    # Log-prob perplexity version: PPL = exp(-1/N * sum(log p(x_i)))
    ppls = [np.exp(-np.mean(np.log(np.maximum(probs, 1e-12)))) for probs in all_per_residue_probs if len(probs) > 0]
    mean_ppl = np.mean(ppls) if len(ppls) > 0 else -1.0
    std_ppl = np.std(ppls) if len(ppls) > 0 else -1.0
    
    return mean_ppl, std_ppl

def eval(d3pm, N, max_length, num_samples=16):
    d3pm.eval()
    with torch.no_grad():
        init_noise = torch.randint(0, N, (num_samples, max_length)).cuda()
        cond = torch.ones((num_samples, 2), device=init_noise.device)
        outputs = d3pm.sample(init_noise, cond)
        
        gen_outputs = []
        sequences = []
        macrel_amp_probs = []
        macrel_hemo_probs = []
        for _i in range(num_samples):
            sent = outputs[_i].cpu().tolist()
            sent = "".join([chr(i + ord('A') - 1) for i in sent])
            clean_sent = sent
            if '@' in sent:
                clean_sent = sent[:sent.index('@')]
            try:
                macrel_amp_prob, macrel_hemo_prob = run_macrel(clean_sent)
            except Exception as e:
                macrel_amp_prob = -1.0
                macrel_hemo_prob = -1.0

            macrel_amp_probs.append(macrel_amp_prob)
            macrel_hemo_probs.append(macrel_hemo_prob)    
            gen_outputs.append(f"[{_i}] {clean_sent} (AMP: {macrel_amp_prob}, Hemo: {macrel_hemo_prob})")
            sequences.append(clean_sent)
        try:
            mean_ppl, std_ppl = ppl(sequences)
        except Exception as e:
            mean_ppl = -1.0
            std_ppl = -1.0
            print(f"Error running perplexity: {e}")

        wandb.log({"generated_text": wandb.Html("<br>".join(gen_outputs)), "macrel_amp_prob": np.mean(macrel_amp_probs), "macrel_hemo_prob": np.mean(macrel_hemo_probs), "mean_ppl": mean_ppl, "std_ppl": std_ppl})

def sample_amps(d3pm, ckpt_path, N, max_length, num_samples, batch_size, output_path):
    state_dict = torch.load(ckpt_path, map_location="cuda")
    d3pm.load_state_dict(state_dict)
    d3pm.eval()

    all_sequences = []
    with torch.no_grad():
        while len(all_sequences) < num_samples:
            current_batch = min(batch_size, num_samples - len(all_sequences))
            init_noise = torch.randint(0, N, (current_batch, max_length), device=next(d3pm.parameters()).device)
            cond = torch.ones((current_batch, 2), device=init_noise.device)
            outputs = d3pm.sample(init_noise, cond)

            for i in range(current_batch):
                sent = outputs[i].cpu().tolist()
                sent = "".join([chr(token + ord("A") - 1) for token in sent])
                # Remove padding token rendered as '@'
                clean_sent = sent.split("@", 1)[0]
                all_sequences.append(clean_sent)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, seq in enumerate(all_sequences):
            f.write(f">{idx}\n{seq}\n")

    print(f"Saved {len(all_sequences)} sampled AMPs to {output_path}")

def train_amps(d3pm, config):
    dataset = SeqDataset(data_path=config["data_path"], max_length=config["max_length"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8)
    print(f"Length of dataloader: {len(dataloader)}")
    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=float(config["lr"]))

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optim,
        num_warmup_steps=100,
        num_training_steps=config["num_train_epochs"] * math.ceil(len(dataloader)),
    )

    d3pm.train()
    global_step = 0
    for _ in range(config["num_train_epochs"]):
        pbar = tqdm(dataloader)
        loss_ema = None
        for x in pbar:
            optim.zero_grad()
            input_ids = x["input_ids"].to(config["device"])
            cond = torch.stack([x["cond_amp"], x["cond_hemo"]], dim=1).to(config["device"])

            loss, info = d3pm(input_ids, cond=cond)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(d3pm.x0_model.parameters(), 5.0)

            with torch.no_grad():
                param_norm = sum([torch.norm(p) for p in d3pm.x0_model.parameters()])

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()

            if global_step % 10 == 0:
                wandb.log({"train_loss": loss, "train_grad_norm": norm, "train_param_norm": param_norm})

            pbar.set_description(
                f"loss: {loss_ema:.4f}, norm: {norm:.4f}, param_norm: {param_norm:.4f}, "
                f"vb_loss: {info['vb_loss']:.4f}, ce_loss: {info['ce_loss']:.4f}"
            )

            optim.step()
            lr_scheduler.step()
            global_step += 1

            if global_step % config["eval_every"] == 1:
                eval(d3pm, config["N"], config["max_length"], num_samples=32)
                d3pm.train()

        torch.save(d3pm.state_dict(), config["checkpoint_path"])

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(config['save_dir'], exist_ok=True)
    wandb.init(project=config['wandb_project'])

    x0_model = DDiT_Llama(N=config["N"])
    d3pm = D3PM(x0_model, config["n_T"], num_classes=config["N"], hybrid_loss_coeff=config["hybrid_loss_coeff"]).to(config["device"])

    print(f"Total Param Count: {sum([p.numel() for p in d3pm.x0_model.parameters()])}")
    if config["mode"] == "train":
        train_amps(d3pm=d3pm, config=config)
    elif config["mode"] == "sample":
        sample_amps(
            d3pm=d3pm,
            ckpt_path=config['checkpoint_path'],
            N=config["N"],
            max_length=config["max_length"],
            num_samples=config["num_samples"],
            batch_size=config["batch_size"],
            output_path=config["output_path"],
        )
    else:
        raise ValueError(f"Unknown mode: {config['mode']}. Use 'train' or 'sample'.")
