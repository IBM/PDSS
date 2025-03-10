# %%
"""
Get data
"""

import pickle
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# %% tags=["parameters"]
upstream = None
product = None
model = None
dataset = None
size = None
# %%
datasets = {
    "molformer-bbbp_finetune": {
        "abnormal": "molecule_data/molformer_finetune_bbbp_embeddings_tiny_class_0.npy",
        "bg": "molecule_data/molformer_finetune_bbbp_embeddings_tiny_class_1.npy",
    }
}
key = "{}-{}".format(model, dataset)
print(key)
bg = np.load(datasets[key]["bg"], allow_pickle=True)
abnormal = np.load(datasets[key]["abnormal"], allow_pickle=True)
print(bg.shape, abnormal.shape)
clean = bg[:size]
print(clean.shape)

# %%
Path("output").mkdir(exist_ok=True)

Path(product["abnormal"]).parent.mkdir(exist_ok=True, parents=True)
Path(product["abnormal"]).write_bytes(pickle.dumps(abnormal))

Path(product["bg"]).parent.mkdir(exist_ok=True, parents=True)
Path(product["bg"]).write_bytes(pickle.dumps(bg[size:]))

Path(product["clean"]).parent.mkdir(exist_ok=True, parents=True)
Path(product["clean"]).write_bytes(pickle.dumps(clean))
# %%
