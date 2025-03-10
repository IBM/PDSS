# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
# ---

# %% tags=["soorgeon-imports"]
from util.sampler import Sampler
from util.pvalranges_calculator import PvalueCalculator
from util.utils import scan_write_metrics, customsort
from tqdm import tqdm
from multiprocessing import Pool
import pickle
from pathlib import Path

# %% tags=["parameters"]
upstream = ["load"]
product = None
scoring = None
typerun = None

# %% tags=["soorgeon-unpickle"]
bg = pickle.loads(Path(upstream["load"]["bg"]).read_bytes())
abnormal = pickle.loads(Path(upstream["load"]["abnormal"]).read_bytes())
clean = pickle.loads(Path(upstream["load"]["clean"]).read_bytes())
print(bg.shape, clean.shape, abnormal.shape)
# %% [markdown]
# ## Run subset scanning

# %%
runs = {
    "group": {
        "clean": {"clean_ssize": 200, "anom_ssize": 0},
        "abnormal": {"clean_ssize": 250, "anom_ssize": 50},
    },
    "individual": {
        "clean": {"clean_ssize": 1, "anom_ssize": 0},
        "abnormal": {"clean_ssize": 0, "anom_ssize": 1},
    },
}

run = 100

for key in ["clean", "abnormal"]:
    run_indices = list()
    print("Run for key: {}".format(key))
    clean_ssize = runs[typerun][key]["clean_ssize"]
    anom_ssize = runs[typerun][key]["anom_ssize"]
    if (clean_ssize != 1 and typerun == "group") or (
        clean_ssize == 1 and typerun == "individual"
    ):
        resultsfile = Path(product["clean_output"])
    if anom_ssize != 0:
        resultsfile = Path(product["adv_output"])

    bg = customsort(bg, conditional=False)
    pvalcalculator = PvalueCalculator(bg)

    records_pvalue_ranges = pvalcalculator.get_pvalue_ranges(clean, pvaltest="1tail")
    anom_records_pvalue_ranges = pvalcalculator.get_pvalue_ranges(
        abnormal, pvaltest="1tail"
    )

    if anom_ssize == 1 and clean_ssize == 0:
        run = anom_records_pvalue_ranges.shape[0]

    elif clean_ssize == 1 and anom_ssize == 0:
        run = records_pvalue_ranges.shape[0]

    samples, sampled_indices = Sampler.sample(
        records_pvalue_ranges,
        anom_records_pvalue_ranges,
        clean_ssize,
        anom_ssize,
        run,
        conditional=False,
    )
    run_indices.append(sampled_indices)

    pool = Pool(processes=5)
    calls = []

    for r_indx in range(run):
        pred_classes = None
        run_sampled_indices = None
        sampled_indices = None

        calls.append(
            pool.apply_async(
                scan_write_metrics,
                [
                    samples[r_indx],
                    pred_classes,
                    clean_ssize,
                    anom_ssize,
                    resultsfile,
                    1,
                    False,
                    None,
                    scoring,
                    -1,
                    run_sampled_indices,
                ],
            )
        )

    print("Beginning Scanning...")
    for sample in tqdm(calls):
        sample.get()

# %%
Path("output").mkdir(exist_ok=True)
Path(product["indices"]).parent.mkdir(exist_ok=True, parents=True)
Path(product["indices"]).write_bytes(pickle.dumps(run_indices))
