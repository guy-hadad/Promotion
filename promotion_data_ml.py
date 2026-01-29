import os, glob
import torch
from recbole.quick_start import run_recbole, load_data_and_model
import recbole.utils.utils as utils_mod
import recbole.trainer.trainer as trainer_mod
import torch.distributed as dist
import numpy as np

MODEL = "SASRec"
DATASETS = ["ml-10m", "ml-100k"] #https://github.com/RUCAIBox/RecBole/blob/master/recbole/properties/dataset/url.yaml
SAVED_DIR = "saved"
OUT_DIR = "scores_memmap"
os.makedirs(OUT_DIR, exist_ok=True)
DTYPE = np.float16

CONFIG_BASE = {
    "TIME_FIELD": "timestamp",
    "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
    "eval_args": {"mode": "full", "order": "TO"},
    "loss_type": "CE",
    "train_neg_sample_args": None,
    "metrics": ["Recall", "NDCG", "MRR", "Hit", "Precision"],
    "topk": [10],
    "show_progress": False,
}


def early_stopping_strict(value, best, cur_step, max_step, bigger=True):
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:  # STRICT (was >=)
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:  # STRICT (was <=)
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def safe_barrier(*args, **kwargs):
    if dist.is_available() and dist.is_initialized():
        return _orig_barrier(*args, **kwargs)
    return None


def latest_ckpt_for(dataset_name: str, saved_dir: str = SAVED_DIR, model_name: str = MODEL) -> str:
    patterns = [
        os.path.join(saved_dir, f"*{model_name}*{dataset_name}*.pth"),
        os.path.join(saved_dir, f"*{dataset_name}*{model_name}*.pth"),
        os.path.join(saved_dir, f"*{dataset_name}*.pth"),
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for dataset={dataset_name} under {saved_dir}/")
    return max(candidates, key=os.path.getmtime)


def infer_num_rows(test_data) -> int:
    """
    Prefer len(test_data.dataset) if available; otherwise count by iterating once.
    Counting once is slower but still does not store scores in RAM.
    """
    ds = getattr(test_data, "dataset", None)
    if ds is not None:
        try:
            return len(ds)
        except Exception:
            pass

    n = 0
    for batch in test_data:
        interaction = batch[0] if isinstance(batch, (tuple, list)) else batch
        n += len(interaction)
    return n


def latest_ckpt_for(dataset_name: str, saved_dir: str = SAVED_DIR) -> str:
    ckpt_dir = os.path.join(saved_dir, dataset_name)
    candidates = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}/ (did training save there?)")
    return max(candidates, key=os.path.getmtime)


utils_mod.early_stopping = early_stopping_strict
trainer_mod.early_stopping = early_stopping_strict
_orig_barrier = dist.barrier
dist.barrier = safe_barrier


for ds_name in DATASETS:
    print(f"\n==============================")
    print(f"Dataset: {ds_name}")
    print(f"==============================")

    ckpt_dir = os.path.abspath(os.path.join(SAVED_DIR, ds_name))
    os.makedirs(ckpt_dir, exist_ok=True)

    cfg = dict(CONFIG_BASE)
    cfg["checkpoint_dir"] = ckpt_dir  # <-- critical fix

    run_recbole(
        model=MODEL,
        dataset=ds_name,
        config_dict=cfg,
        saved=True,
    )
    ckpt = latest_ckpt_for(ds_name)
    print("Loaded checkpoint:", ckpt)

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(ckpt)

    device = config["device"]
    model = model.to(device).eval()

    item_num = getattr(dataset, "item_num", None)
    if item_num is None:
        raise RuntimeError("Could not read dataset.item_num from RecBole dataset object.")
    print("item_num:", item_num)

    N = infer_num_rows(test_data)
    print("N test rows:", N)

    approx_gb = (N * item_num * np.dtype(DTYPE).itemsize) / (1024**3)
    print(f"Approx file size: {approx_gb:.2f} GB")

    out_path = os.path.join(OUT_DIR, f"{MODEL}_{ds_name}_full_scores_float16.npy")
    scores_mm = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=DTYPE, shape=(N, item_num)
    )
    print("Writing full scores to:", out_path)

    row_offset = 0
    with torch.no_grad():
        for batch in test_data:
            # (interaction, (history_u, history_i), positive_u, positive_i)
            if isinstance(batch, (tuple, list)) and len(batch) == 4:
                interaction, history_index, positive_u, positive_i = batch
            else:
                interaction = batch[0] if isinstance(batch, (tuple, list)) else batch
                history_index = None

            interaction = interaction.to(device)

            # [B, item_num]
            s = model.full_sort_predict(interaction)

            if s.dim() == 1:
                s = s.unsqueeze(0)

            # mask PAD item 0
            s[:, 0] = float("-inf")

            if history_index is not None:
                if isinstance(history_index, (tuple, list)) and len(history_index) == 2:
                    history_u, history_i = history_index
                    s[history_u.to(device), history_i.to(device)] = float("-inf")
                else:
                    # fallback if some other loader returns flat indices
                    s.view(-1).index_fill_(0, history_index.to(device), float("-inf"))

            B = s.size(0)

            scores_mm[row_offset:row_offset + B] = (
                s.detach().cpu().numpy().astype(DTYPE, copy=False)
            )
            row_offset += B


    scores_mm.flush()
    print(f"Done. Saved memmap scores for {ds_name} with shape {(N, item_num)}")

    del model
    torch.cuda.empty_cache()

print("\nALL DONE.")
