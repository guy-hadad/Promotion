import os
import glob
import numpy as np
import torch
from tqdm.auto import tqdm
from recbole.quick_start import run_recbole, load_data_and_model
import torch.distributed as dist

def safe_barrier(*args, **kwargs):
    if dist.is_available() and dist.is_initialized():
        return _orig_barrier(*args, **kwargs)
    return None


_orig_barrier = dist.barrier
dist.barrier = safe_barrier

MODEL = "SASRec"
DATASETS = ["steam-merged"]          # add more if you want
DATA_PATH = "dataset"               # where RecBole datasets folder exists
SAVED_DIR = "saved"                 # RecBole checkpoints will go here
OUT_DIR = "scores_memmap"           # memmap output
os.makedirs(OUT_DIR, exist_ok=True)

DTYPE_SCORES = np.float16
DTYPE_IDS = np.int32

SAVE_USER_IDS = True
SAVE_POS_ITEM = True


CONFIG_BASE = {
    # data
    "data_path": DATA_PATH,
    "load_col": {"inter": ["user_id", "product_id", "timestamp"]},
    "USER_ID_FIELD": "user_id",
    "ITEM_ID_FIELD": "product_id",
    "TIME_FIELD": "timestamp",

    "eval_args": {"mode": "full", "order": "TO"},  # full-sort eval
    "train_neg_sample_args": None,
    "loss_type": "CE",

    "metrics": ["Recall", "NDCG", "MRR", "Hit", "Precision"],
    "topk": [10],
    "valid_metric": "Recall@10",
    "valid_metric_bigger": True,

    "reproducibility": True,
    "show_progress": False,
    "log_wandb": False,
}


def latest_ckpt_for(dataset_name: str, saved_dir: str = SAVED_DIR) -> str:
    """
    RecBole saves checkpoints under:
      saved/<dataset_name>/*.pth
    """
    ckpt_dir = os.path.join(saved_dir, dataset_name)
    candidates = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in {ckpt_dir}/. "
            f"(Did you run training with saved=True and checkpoint_dir set?)"
        )
    return max(candidates, key=os.path.getmtime)


def infer_num_rows(test_data) -> int:
    """
    Prefer len(test_data.dataset) if possible; otherwise count by iterating once.
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


def parse_fullsort_batch(batch):
    """
    FullSortEvalDataLoader returns usually:
      (interaction, (history_u, history_i), positive_u, positive_i)

    But some versions may differ slightly.
    """
    if isinstance(batch, (tuple, list)) and len(batch) == 4:
        return batch  # (interaction, history_index, pos_u, pos_i)
    interaction = batch[0] if isinstance(batch, (tuple, list)) else batch
    return interaction, None, None, None



for ds_name in DATASETS:
    print("\n==============================")
    print(f"Training from scratch: {MODEL} on {ds_name}")
    print("==============================")

    ckpt_dir = os.path.abspath(os.path.join(SAVED_DIR, ds_name))
    os.makedirs(ckpt_dir, exist_ok=True)

    cfg = dict(CONFIG_BASE)
    cfg["checkpoint_dir"] = ckpt_dir  # critical

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

    approx_gb = (N * item_num * np.dtype(DTYPE_SCORES).itemsize) / (1024**3)
    print(f"Approx scores file size (float16): {approx_gb:.2f} GB")

    scores_path = os.path.join(OUT_DIR, f"{MODEL}_{ds_name}_test_full_scores_f16.npy")
    scores_mm = np.lib.format.open_memmap(
        scores_path, mode="w+", dtype=DTYPE_SCORES, shape=(N, item_num)
    )
    print("Writing scores to:", scores_path)

    uid_field = getattr(dataset, "uid_field", None)  # usually "user_id"

    user_ids_mm = None
    if SAVE_USER_IDS and uid_field is not None:
        user_ids_path = os.path.join(OUT_DIR, f"{MODEL}_{ds_name}_test_user_ids_i32.npy")
        user_ids_mm = np.lib.format.open_memmap(
            user_ids_path, mode="w+", dtype=DTYPE_IDS, shape=(N,)
        )
        print("Writing user_ids to:", user_ids_path)

    pos_item_mm = None
    if SAVE_POS_ITEM:
        pos_path = os.path.join(OUT_DIR, f"{MODEL}_{ds_name}_test_pos_item_i32.npy")
        pos_item_mm = np.lib.format.open_memmap(
            pos_path, mode="w+", dtype=DTYPE_IDS, shape=(N,)
        )
        print("Writing pos_item to:", pos_path)

    row_offset = 0
    with torch.no_grad():
        for batch in tqdm(test_data, desc=f"Dumping full-sort scores ({ds_name})", leave=True):
            interaction, history_index, pos_u, pos_i = parse_fullsort_batch(batch)
            interaction = interaction.to(device)

            # scores [B, item_num]
            s = model.full_sort_predict(interaction)
            if s.dim() == 1:
                s = s.unsqueeze(0)

            B = s.size(0)

            # mask padding item 0
            s[:, 0] = -float("inf")

            # mask history items if provided
            if history_index is not None and isinstance(history_index, (tuple, list)) and len(history_index) == 2:
                history_u, history_i = history_index
                s[history_u.to(device), history_i.to(device)] = -float("inf")

            if user_ids_mm is not None and uid_field is not None:
                uids = interaction[uid_field].detach().cpu().numpy().astype(DTYPE_IDS, copy=False)
                user_ids_mm[row_offset:row_offset + B] = uids

            if pos_item_mm is not None and pos_i is not None:
                pos_item_mm[row_offset:row_offset + B] = (
                    pos_i.detach().cpu().numpy().astype(DTYPE_IDS, copy=False)
                )

            s = s.to(torch.float16)
            scores_mm[row_offset:row_offset + B] = s.detach().cpu().numpy()

            row_offset += B

    scores_mm.flush()
    if user_ids_mm is not None:
        user_ids_mm.flush()
    if pos_item_mm is not None:
        pos_item_mm.flush()

    print(f"Done. Saved scores with shape: {(row_offset, item_num)}")

    # free GPU
    del model
    torch.cuda.empty_cache()

print("\nALL DONE.")
