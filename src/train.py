"""Reproducible fixed-dataset training, with a fresh deterministic heldout panel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from data import Task
from model import ClauseTransformer, ModelConfig, load_model


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


@torch.no_grad()
def quick_eval(model, task, length, n=256):
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(99001 + length)
    sample = task.generate(n, length, gen, device=device)
    clauses = sample["clauses"]
    preds = model(clauses[:, :-1])[:, length:2*length].argmax(-1)
    targets = clauses[:, length+1:]
    val = preds[:, :, 4].eq(targets[:, :, 4])
    whole = preds.eq(targets).all(-1)
    return {"length": length, "value_step": val.float().mean().item(),
            "value_chain": val.all(-1).float().mean().item(),
            "clause_chain": whole.all(-1).float().mean().item(),
            "variable_step": preds[:, :, 3].eq(targets[:, :, 3]).float().mean().item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="C6")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--length", type=int, default=5)
    p.add_argument("--nvars", type=int, default=256)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--samples", type=int, default=8192)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--token-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--pseudo-data", default=None)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--no-residual", action="store_true")
    p.add_argument("--no-layer-norm", action="store_true")
    p.add_argument("--activation", default="gelu")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = "cuda"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    task = Task(args.task, nvars=args.nvars)
    if args.resume:
        model, _ = load_model(args.resume, device)
    else:
        model = ClauseTransformer(ModelConfig(task.vocab_size, token_dim=args.token_dim,
            residual=not args.no_residual, layer_norm=not args.no_layer_norm,
            activation=args.activation)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(10000 + args.seed)
    if args.pseudo_data:
        data = torch.load(args.pseudo_data, map_location="cpu", weights_only=False)["clauses"].to(device)
    else:
        data = task.generate(args.samples, args.length, gen, device=device)["clauses"]
    args.samples = len(data)
    train_hash = hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest()
    source_hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")}
    write_json(out / "config.json", {**vars(args), "model": model.config_dict(),
        "dataset_sha256": train_hash, "source_sha256": source_hashes,
        "parameters": sum(x.numel() for x in model.parameters()),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "epoch_definition": "one pass through a fixed independently generated training dataset"})
    start = time.monotonic()
    def save(epoch):
        path = out / f"checkpoint-{epoch:04d}.pt"
        torch.save({"model": model.state_dict(), "model_config": model.config_dict(),
                    "args": vars(args), "epoch": epoch, "dataset_sha256": train_hash}, path)
    def evaluate(epoch, loss, slot_loss=None):
        model.eval()
        panel = [quick_eval(model, task, length) for length in [5, 10, 20, 40]]
        row = {"epoch": epoch, "loss": loss, "slot_loss": slot_loss, "seconds": time.monotonic()-start, "eval": panel}
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        write_json(out / "progress.json", row)
        print(json.dumps(row), flush=True)
    save(0)
    evaluate(0, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(data), device=device)
        losses = []
        slots = []
        for indices in permutation.split(args.batch):
            batch = data[indices]
            logits = model(batch[:, :-1])[:, args.length:2*args.length]
            targets = batch[:, args.length+1:]
            # Mean over samples, CoT positions and all five slots; same minimum
            # as Eq6, different constant loss normalization explicitly recorded.
            loss = F.cross_entropy(logits.reshape(-1, task.vocab_size), targets.reshape(-1))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            losses.append(loss.detach())
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                with torch.no_grad():
                    slots.append(F.cross_entropy(logits.reshape(-1, task.vocab_size),
                                 targets.reshape(-1), reduction="none").reshape(-1, 5).mean(0))
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            evaluate(epoch, torch.stack(losses).mean().item(), torch.stack(slots).mean(0).tolist())
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs or epoch in [5,10,15,20]:
            save(epoch)
    write_json(out / "complete.json", {"epochs": args.epochs,"seconds":time.monotonic()-start,
               "checkpoint":str(out / f"checkpoint-{args.epochs:04d}.pt")})


if __name__ == "__main__":
    main()
