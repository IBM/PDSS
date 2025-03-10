# %% tags=["parameters"]
upstream = ["run_ss"]
product = None
model = None
dataset = None
legendclean = None
legendalt = None
# %%
from pathlib import Path
import pickle

abnormal_path = Path(upstream["run_ss"]["adv_output"])
clean_path = Path(upstream["run_ss"]["clean_output"])
indices = pickle.loads(Path(upstream["run_ss"]["indices"]).read_bytes())


# %%
print(indices)

# %% tags=["soorgeon-imports"]
import matplotlib.pyplot as plt
from util.resultparser import ResultParser, ResultSelector


def load_results(clean_fn, fake_fn):
    resultselector = ResultSelector(score=True)
    a = ResultParser.get_results(clean_fn, resultselector)
    b = ResultParser.get_results(fake_fn, resultselector)
    clean_scores = np.array(a["scores"])
    anom_scores = np.array(b["scores"])

    return clean_scores, anom_scores


# %%
import seaborn as sns
from sklearn import metrics
import numpy as np

plt.rcParams.update({"font.size": 11})
pal = sns.color_palette("Set2")

sns.set_palette(pal)


def plot_layer(model, clean_fn, fake_fn):
    """
    Load PDSS scores for each run and display score distribution and AUC
    """
    clean_scores, anom_scores = load_results(clean_fn=clean_fn, fake_fn=fake_fn)
    clean_scores = clean_scores[~np.isinf(clean_scores)]
    anom_scores = anom_scores[~np.isinf(anom_scores)]

    sns.set(style="darkgrid")
    sns.histplot(
        clean_scores,
        element="step",
        stat="density",
        kde=True,
        color="green",
        label=legendclean,
    )
    sns.histplot(
        anom_scores,
        element="step",
        stat="density",
        kde=True,
        color="purple",
        label=legendalt,
    )

    plt.title("Distribution of subset scores for - {} - {}".format(model, dataset))
    plt.legend()
    plt.ylabel("Density")
    plt.xlabel("Subset Score")
    y_true = np.append([np.ones(len(anom_scores))], [np.zeros(len(clean_scores))])
    all_scores = np.append([anom_scores], [clean_scores])

    fpr, tpr, thresholds = metrics.roc_curve(y_true, all_scores)
    roc_auc = metrics.auc(fpr, tpr)
    print("AUC: {}".format(roc_auc))
    plt.show()


# %%
plot_layer(model, clean_path, abnormal_path)


# %%


def get_anom_nodes(fn="results/"):
    """
    Extract from output file the list of elements that were found anom.
    remove end of line and return an int list of positions.
    """
    nodes_samples = []
    images_samples = []
    # precision_samples = []
    # recall_samples = []
    optimal_alpha_samples = []
    with open(fn, "r") as f:
        for line in f.readlines()[:-1]:
            # print(line.rstrip().split(' '))
            if "inf" in line.rstrip().split(" "):
                pass
            else:
                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    optimal_alpha,
                    anom_node,
                    anom_images,
                ) = line.rstrip().split(" ")
                nodes = anom_node.strip().split(",")
                images = anom_images.strip().split(",")
                if nodes != [""]:
                    nodes = list(map(int, nodes))
                images = list(map(int, images))
                nodes_samples.append(np.array(nodes))
                images_samples.append(images)
                optimal_alpha_samples.append(float(optimal_alpha))

    return (
        nodes_samples,
        images_samples,
        _,
        _,
        optimal_alpha_samples,
    )


# %%

nodes_fake, _, _, _, optimal_alpha_samples_fake = get_anom_nodes(abnormal_path)
nodes_clean, _, _, _, optimal_alpha_samples_clean = get_anom_nodes(clean_path)

optimal_alpha_samples_fake = np.array(optimal_alpha_samples_fake)
optimal_alpha_samples_clean = np.array(optimal_alpha_samples_clean)
nodes_fake = np.array([len(n) for n in nodes_fake])
plt.title("Alpha distribution for  - {} - {}".format(model, dataset))
sns.histplot(
    optimal_alpha_samples_fake,
    element="step",
    stat="density",
    kde=True,
    color="purple",
    label="$H_1$",
)
sns.histplot(
    optimal_alpha_samples_clean,
    element="step",
    stat="density",
    kde=True,
    color="green",
    label="$H_0$",
)
plt.xlabel("Alpha values")
plt.ylabel("Density")
plt.legend()
plt.show()
plt.title(
    "Subset cardinality distribution for samples   - {} - {}".format(model, dataset)
)
plt.xlabel("Subset cardinality values")
plt.ylabel("Density")
sns.distplot(nodes_fake, label=legendalt, color="purple")
plt.legend()
plt.show()


# %%
import matplotlib
import matplotlib.patches as mpatches
import seaborn as sb
import matplotlib.ticker as ticker

nodes_fake, molecule_samples, precision, recall, optimal_alpha_samples_fake = (
    get_anom_nodes(abnormal_path)
)


def draw_anoms(nodes, class_target="", shape=(1, 786)):
    """
    Mark as 1 anomalous elements in the shape parameter,
    rest of the nodes is mark as 0.
    """
    zeros = np.zeros(shape[1])
    np.put(zeros, nodes, 1)
    return zeros.reshape(shape)


all_viz = list()
for i, node in enumerate(nodes_fake):
    # if precision[i] >= 0.75:
    all_viz.append(draw_anoms(node))
palet = sb.cubehelix_palette(8)
plt.figure(figsize=(40, 40))
plt.title("Anom nodes in embedding - {} - {}".format(model, dataset))
heat_map = sb.heatmap(
    np.array(all_viz).reshape(-1, 786),
    fmt="",
    square=True,
    cbar_kws={"shrink": 0.5},
    cmap=palet,
    cbar=False,
)
heat_map.xaxis.set_major_locator(ticker.MultipleLocator(10))
heat_map.xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.tight_layout()
plt.show()

# %%
from collections import Counter

flat_nodes = [node for run in nodes_fake for node in run]
flat_nodes_count = Counter(flat_nodes)
values, counts = zip(*flat_nodes_count.most_common(100))

# sb.set(style="darkgrid")
plt.figure(figsize=(20, 3))
plt.title("Anom nodes in embedding - {} - {}".format(model, dataset))
heat_map = sb.heatmap(
    draw_anoms(values).reshape(-1, 786),
    fmt="",
    square=False,
    cbar_kws={"shrink": 0.5},
    cbar=False,
)
heat_map.xaxis.set_major_locator(ticker.MultipleLocator(10))
heat_map.xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.tight_layout()
plt.show()
