import math
import os
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from Bio import SeqIO
from tqdm import tqdm
from transformers import get_scheduler
import wandb
import torch
import numpy as np

from d3pm import D3PM
from dit import DDiT_Llama
from macrel.run_macrel import run_macrel

# Avoid tokenizers fork-parallelism warning/deadlock risk with DataLoader workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def get_output_path(config):
    run_name = get_run_name(config)
    return os.path.join("samples", f"sampled_amps_{run_name}.fasta")

def get_save_path(config):
    run_name = get_run_name(config)
    return os.path.join("ckpt", f"model_{run_name}.pth")

def get_run_name(config):
    # Build a stable, filesystem-friendly tag from key training/model hyperparameters.
    def fmt(value):
        return format(value, ".0e").replace("+", "") if abs(value) < 1e-3 else str(value).replace(".", "p")

    return (
        f"lr{fmt(float(config['lr']))}"
        f"_wd{fmt(float(config['weight_decay']))}"
        f"_hlc{fmt(float(config['hybrid_loss_coeff']))}"
        f"_d{config['dim']}_l{config['n_layers']}_h{config['n_heads']}"
    )

def read_fasta(path):
    records_raw = list(SeqIO.parse(path, "fasta"))
    return [str(x.seq) for x in records_raw]

class SeqDataset(Dataset):
    def __init__(self, data_path, max_length):
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

def eval(d3pm, config):
    d3pm.eval()
    with torch.no_grad():
        init_noise = torch.randint(0, config["N"], (config["num_samples"], config["max_length"])).cuda()
        cond = torch.zeros((config["num_samples"], 2), device=init_noise.device)
        cond[:, 0] = 1
        outputs = d3pm.sample(init_noise, cond)
   
        
        gen_outputs = []
        macrel_amp_probs = []
        macrel_hemo_probs = []
        for _i in range(config["num_samples"]):
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

        wandb.log({"generated_text": wandb.Html("<br>".join(gen_outputs)), "macrel_amp_prob": np.mean(macrel_amp_probs), "macrel_hemo_prob": np.mean(macrel_hemo_probs)})

def train_amps(d3pm, config):
    dataset = SeqDataset(data_path=config["data_path"], max_length=config["max_length"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8)
    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))

    lr_scheduler = get_scheduler(name="linear", optimizer=optim, num_warmup_steps=5000, num_training_steps=config["num_train_epochs"] * math.ceil(len(dataloader)))

    d3pm.train()
    global_step = 0
    for _ in range(config["num_train_epochs"]):
        pbar = tqdm(dataloader)
        loss_ema = None
        for x in pbar:
            optim.zero_grad()
            input_ids = x["input_ids"].to(config["device"])
            cond = torch.stack([x["cond_amp"], x["cond_hemo"]], dim=1).to(config["device"])

            loss = d3pm(input_ids, cond=cond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(d3pm.x0_model.parameters(), 5.0)

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()

            if global_step % 10 == 0:
                wandb.log({"train_loss": loss_ema})

            pbar.set_description(f"loss: {loss_ema:.4f}")

            optim.step()
            lr_scheduler.step()
            global_step += 1

            if global_step % config["eval_every"] == 1:
                eval(d3pm, config)
                d3pm.train()

        torch.save(d3pm.state_dict(), get_save_path(config))

def check_eval(d3pm, config):
    state_dict = torch.load(get_save_path(config), map_location="cuda")
    d3pm.load_state_dict(state_dict)
    d3pm.eval()

    with torch.no_grad():
        init_noise = torch.randint(0, config["N"], (1, config["max_length"]), device=next(d3pm.parameters()).device)
        cond = torch.ones((1, 2), device=init_noise.device)
        outputs = d3pm.sample(init_noise, cond, return_all_samples=True)

        for i, sent in enumerate(outputs):
            sent = sent.cpu().tolist()[0]
            sent = "".join([chr(token + ord("A") - 1) for token in sent])
            # Remove padding token rendered as '@'
            clean_sent = sent.split("@", 1)[0]
            try:
                macrel_amp_prob, macrel_hemo_prob = run_macrel(clean_sent)
            except Exception as e:
                macrel_amp_prob = -1.0
                macrel_hemo_prob = -1.0
            print(f"[{i}] {clean_sent} (AMP: {macrel_amp_prob}, Hemo: {macrel_hemo_prob})")

def sample_amps(d3pm, config):
    state_dict = torch.load(get_save_path(config), map_location="cuda")
    d3pm.load_state_dict(state_dict)
    d3pm.eval()

    all_sequences = []
    with torch.no_grad():
        while len(all_sequences) < config["num_samples"]:
            current_batch = min(config["batch_size"], config["num_samples"] - len(all_sequences))
            init_noise = torch.randint(0, config["N"], (current_batch, config["max_length"]), device=next(d3pm.parameters()).device)
            cond = torch.zeros((current_batch, 2), device=init_noise.device)
            cond[:, 0] = 1
            outputs = d3pm.sample(init_noise, cond)

            for i in range(current_batch):
                sent = outputs[i].cpu().tolist()
                sent = "".join([chr(token + ord("A") - 1) for token in sent])
                # Remove padding token rendered as '@'
                clean_sent = sent.split("@", 1)[0]
                all_sequences.append(clean_sent)

    with open(get_output_path(config), "w", encoding="utf-8") as f:
        for idx, seq in enumerate(all_sequences):
            f.write(f">{idx}\n{seq}\n")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print(config)
    wandb.init(project="d3pm_amp", config=config)

    x0_model = DDiT_Llama(N=config["N"], dim=config["dim"], n_layers=config["n_layers"], n_heads=config["n_heads"])
    print(x0_model)
    print(f"Total Param Count: {sum([p.numel() for p in x0_model.parameters()])}")
    d3pm = D3PM(x0_model, config["n_T"], num_classes=config["N"], hybrid_loss_coeff=config["hybrid_loss_coeff"]).to(config["device"])

    if config["mode"] == "train":
        train_amps(d3pm=d3pm, config=config)
        sample_amps(d3pm=d3pm, config=config)
    elif config["mode"] == "sample":
        sample_amps(d3pm=d3pm, config=config)
    elif config["mode"] == "eval":
        check_eval(d3pm=d3pm, config=config)
    else:
        raise ValueError(f"Unknown mode: {config['mode']}. Use 'train' or 'sample' or 'eval'.")
