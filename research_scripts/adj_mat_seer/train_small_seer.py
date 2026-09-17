"""
CLI twin of train_small_seer.ipynb — prefer the notebook.

Train a small AdjMatSeer on OpenBabel-labelled bonds from generate_obabel_pairs.ipynb
(teacher-pair x1 or 420 EDM @ 100 steps — not FM).

Data: ./bond_pairs/shard*.pt from generate_obabel_pairs.ipynb
  elements (int8, DIMENSION)     atomic numbers, canonical order
  coords   (f16, DIMENSION, 3)   canonical order, Angstrom
  conn     (int8, D, D)          RDKit DetermineConnectivity guess  -> model INPUT
  target   (int8, D, D)          OpenBabel bond orders 0-4          -> model TARGET
  n_atoms  (int16)

Inputs are rebuilt to match `prepare_adj_mat_seer_input` exactly: dist_mat + I, and
binary (conn + I) clamped to 1. Nothing here may use `target` to build an input.

Selection metric is molecule VALIDITY (RDKit sanitization of the reconstructed graph),
not the loss - that is the number the README quotes as 48% (ML) vs 93% (OpenBabel).

    python train_small_seer.py --n-hidden 512 --epochs 30
    python train_small_seer.py --n-hidden 256 --epochs 40 --baseline ../../adj_mat_seer_chembl_15_39.pt

The 2048-wide baseline is ~21.8 M params / 83 MB; 512 -> ~1.6 M / 6 MB, 256 -> ~0.5 M / 2 MB.
Seer compute is only ~5% of an 8-NFE sample, so expect a packaging win, not a speed win.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rdkit import Chem, RDLogger  # noqa: E402

from src.mlconfgen.adj_mat_seer import AdjMatSeer  # noqa: E402
from src.mlconfgen.utils.common import bond_type_dict  # noqa: E402
from src.mlconfgen.utils.config import DIMENSION, NUM_BOND_TYPES  # noqa: E402

RDLogger.DisableLog("rdApp.*")


# --------------------------------------------------------------------------- data
class BondData:
    """All shards in RAM (int8/f16 -> ~3.6 kB per molecule)."""

    def __init__(self, data_dir: Path, max_shards: int | None, log):
        shards = sorted(data_dir.glob("shard*.pt"))
        if not shards:
            raise FileNotFoundError(f"no shard*.pt in {data_dir}")
        if max_shards:
            shards = shards[:max_shards]
        packs = [torch.load(p, map_location="cpu", weights_only=False) for p in shards]
        self.elements = torch.cat([p["elements"] for p in packs])
        self.coords = torch.cat([p["coords"] for p in packs])
        self.conn = torch.cat([p["conn"] for p in packs])
        self.target = torch.cat([p["target"] for p in packs])
        self.n_atoms = torch.cat([p["n_atoms"] for p in packs])
        self.aromatic_mode = packs[0].get("aromatic_mode", "?")
        nbytes = sum(t.numel() * t.element_size() for t in
                     (self.elements, self.coords, self.conn, self.target))
        log(f"loaded {len(shards)} shards  n={self.n_atoms.shape[0]} "
            f"arom={self.aromatic_mode}  ram={nbytes / 1e9:.2f} GB")

    def __len__(self):
        return self.n_atoms.shape[0]


def build_inputs(d: BondData, idx: torch.Tensor, device: str):
    """-> elements(long), dist_mat(float), adj_in(float), target(long), n_atoms(long)"""
    elements = d.elements[idx].to(device=device, dtype=torch.long)
    coords = d.coords[idx].to(device=device, dtype=torch.float32)
    conn = d.conn[idx].to(device=device, dtype=torch.float32)
    target = d.target[idx].to(device=device, dtype=torch.long)
    n_atoms = d.n_atoms[idx].to(device=device, dtype=torch.long)

    eye = torch.eye(DIMENSION, device=device)
    # padded rows are zero in coords, so zero them in the distance matrix too
    valid = (torch.arange(DIMENSION, device=device)[None, :] < n_atoms[:, None]).float()
    pair_valid = valid[:, :, None] * valid[:, None, :]
    dist = torch.cdist(coords, coords) * pair_valid + eye
    adj_in = (conn + eye).clamp(max=1.0)
    return elements, dist, adj_in, target, n_atoms


def pair_mask(n_atoms: torch.Tensor, device: str) -> torch.Tensor:
    """Upper triangle of the real n x n block; the model output is symmetric."""
    ar = torch.arange(DIMENSION, device=device)
    valid = ar[None, :] < n_atoms[:, None]
    both = valid[:, :, None] & valid[:, None, :]
    return both & (ar[None, :, None] < ar[None, None, :])


# --------------------------------------------------------------------------- loss
def class_weights(d: BondData, n_sample: int, cap: float, device: str) -> torch.Tensor:
    """Inverse sqrt-frequency, capped. ~94% of pairs are 'no bond'."""
    idx = torch.arange(min(n_sample, len(d)))
    tgt = d.target[idx].to(torch.long)
    na = d.n_atoms[idx].to(torch.long)
    m = pair_mask(na, "cpu")
    vals = tgt[m]
    counts = torch.bincount(vals, minlength=NUM_BOND_TYPES).float().clamp_min(1.0)
    w = (counts.sum() / counts).sqrt()
    w = (w / w[0]).clamp(max=cap)
    return w.to(device)


# --------------------------------------------------------------------------- eval
def mol_is_valid(elements: np.ndarray, orders: np.ndarray, n: int) -> bool:
    m = Chem.RWMol()
    for z in elements[:n]:
        m.AddAtom(Chem.Atom(int(z)))
    for i in range(n):
        for j in range(i):
            o = int(orders[i, j])
            if o:
                m.AddBond(j, i, bond_type_dict[o])
    try:
        Chem.SanitizeMol(m.GetMol())
        return True
    except Exception:
        return False


@torch.inference_mode()
def evaluate(model, d: BondData, idx: torch.Tensor, device: str, batch: int,
             weights: torch.Tensor, n_validity: int):
    """CE, bond P/R/F1, exact-graph match, and molecule validity rate."""
    model.eval()
    ce_sum = n_pairs = 0.0
    tp = fp = fn = cls_ok = cls_tot = 0
    exact = n_mol = 0
    valid = checked = 0

    for s in range(0, idx.shape[0], batch):
        bidx = idx[s : s + batch]
        el, dist, adj_in, tgt, na = build_inputs(d, bidx, device)
        logits = model(elements=el, dist_mat=dist, adj_mat=adj_in)
        m = pair_mask(na, device)

        lg, tg = logits[m], tgt[m]
        ce_sum += F.cross_entropy(lg, tg, weight=weights, reduction="sum").item()
        n_pairs += tg.numel()

        pred = lg.argmax(-1)
        pb, tb = pred > 0, tg > 0
        tp += (pb & tb).sum().item()
        fp += (pb & ~tb).sum().item()
        fn += (~pb & tb).sum().item()
        cls_ok += (pred[tb] == tg[tb]).sum().item()
        cls_tot += int(tb.sum().item())

        full = logits.argmax(-1)
        wrong = ((full != tgt) & m).flatten(1).any(1)
        exact += int((~wrong).sum().item())
        n_mol += bidx.shape[0]

        if checked < n_validity:
            take = min(bidx.shape[0], n_validity - checked)
            orders = full.cpu().numpy()
            els = d.elements[bidx[:take]].numpy()
            nas = d.n_atoms[bidx[:take]].numpy()
            for k in range(take):
                valid += mol_is_valid(els[k], orders[k], int(nas[k]))
            checked += take

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "ce": ce_sum / max(n_pairs, 1),
        "bond_p": prec,
        "bond_r": rec,
        "bond_f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "order_acc": cls_ok / max(cls_tot, 1),
        "exact": exact / max(n_mol, 1),
        "valid": valid / max(checked, 1),
        "n_valid_checked": checked,
    }


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("./bond_pairs"))
    ap.add_argument("--out-dir", type=Path, default=Path("./checkpoints_seer"))
    ap.add_argument("--n-hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--weight-cap", type=float, default=25.0,
                    help="cap on inverse-frequency class weight")
    ap.add_argument("--n-validity", type=int, default=2000,
                    help="molecules per eval for the validity metric")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="2048-wide seer ckpt to score on the same val split. Only its "
                         "'valid' number is comparable: it was trained with aromatic=class 4, "
                         "so against kekule labels its exact/order_acc are meaningless.")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"train_seer_{args.n_hidden}.log"

    def log(msg):
        line = f"{datetime.now().isoformat(timespec='seconds')}  {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    torch.manual_seed(args.seed)
    dev = args.device

    d = BondData(args.data_dir, args.max_shards, log)
    perm = torch.randperm(len(d), generator=torch.Generator().manual_seed(args.seed))
    n_val = max(1, int(len(d) * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    log(f"split train={train_idx.shape[0]} val={val_idx.shape[0]}")

    w = class_weights(d, 20_000, args.weight_cap, dev)
    log("class weights " + " ".join(f"{k}:{v:.2f}" for k, v in enumerate(w.tolist())))

    model = AdjMatSeer(n_hidden=args.n_hidden, device=dev).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"AdjMatSeer n_hidden={args.n_hidden}  params={n_par / 1e6:.2f} M "
        f"({n_par * 4 / 1e6:.1f} MB fp32)")
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    if args.baseline and args.baseline.exists():
        base = AdjMatSeer(n_hidden=2048, device=dev).to(dev)
        bs = torch.load(args.baseline, map_location=dev, weights_only=False)
        base.load_state_dict(bs.get("state_dict", bs))
        bm = evaluate(base, d, val_idx, dev, args.batch, w, args.n_validity)
        log(f"BASELINE 2048  valid={bm['valid']:.3f}  <- the only comparable number "
            f"(README quotes 48% for this model); exact={bm['exact']:.3f} "
            f"bond_f1={bm['bond_f1']:.4f} order_acc={bm['order_acc']:.4f} are vs "
            f"{d.aromatic_mode} labels, so ignore them if mode=kekule")
        del base
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    best_valid, no_imp = -1.0, 0
    for epoch in range(args.epochs):
        model.train()
        ep = train_idx[torch.randperm(train_idx.shape[0])]
        run, ns = 0.0, 0
        pbar = tqdm(range(0, ep.shape[0] - args.batch + 1, args.batch),
                    desc=f"h{args.n_hidden} ep{epoch}")
        for s in pbar:
            bidx = ep[s : s + args.batch]
            el, dist, adj_in, tgt, na = build_inputs(d, bidx, dev)
            logits = model(elements=el, dist_mat=dist, adj_mat=adj_in)
            m = pair_mask(na, dev)
            loss = F.cross_entropy(logits[m], tgt[m], weight=w)
            if not torch.isfinite(loss):
                log(f"WARN non-finite loss ep={epoch}")
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item()
            ns += 1
            if ns % 50 == 0:
                pbar.set_postfix(ce=f"{run / ns:.4f}")
        sched.step()

        mt = evaluate(model, d, val_idx, dev, args.batch, w, args.n_validity)
        log(f"ep={epoch} train_ce={run / max(ns, 1):.4f} val_ce={mt['ce']:.4f} "
            f"valid={mt['valid']:.3f} exact={mt['exact']:.3f} "
            f"bond_f1={mt['bond_f1']:.4f} bond_p={mt['bond_p']:.4f} bond_r={mt['bond_r']:.4f} "
            f"order_acc={mt['order_acc']:.4f}")

        ckpt = {
            "state_dict": model.state_dict(),
            "n_hidden": args.n_hidden,
            "epoch": epoch,
            "metrics": mt,
            "aromatic_mode": d.aromatic_mode,
            "args": vars(args) | {"data_dir": str(args.data_dir),
                                  "out_dir": str(args.out_dir),
                                  "baseline": str(args.baseline)},
        }
        torch.save(ckpt, args.out_dir / f"latest_seer_{args.n_hidden}.pt")
        if mt["valid"] > best_valid:
            best_valid, no_imp = mt["valid"], 0
            torch.save(ckpt, args.out_dir / f"best_seer_{args.n_hidden}.pt")
            log(f"ckpt best valid={best_valid:.3f}")
        else:
            no_imp += 1
            if no_imp >= args.patience:
                log(f"early stop ep={epoch}")
                break

    log(f"done n_hidden={args.n_hidden} best_valid={best_valid:.3f}")


if __name__ == "__main__":
    main()
