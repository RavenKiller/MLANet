import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import time

if not os.path.exists("data/plots"):
    os.mkdir("data/plots")
# model_names = ["mla_sigma0.5","mla_sigma0.8","mla_sigma1.0"]
# model_names = ["mla_drop0.0","mla_drop0.15","mla_drop0.3","mla_drop0.45","mla_drop0.6"]
# model_names = ["mla_type0","mla_type1","mla_type2","mla_type3","mla_nrsub", "mla_fgsub"]
# model_names = ["mla_da"]
# model_names = ["mla_nrsub", "mla_fgsub"]"]
# model_names = ["mla_aug_50d","mla_aug_back","mla_aug3","mla_aug2"]
# model_names = ["mla_50d","mla","mla_type0"]

# model_names = ["mla_aug3","mla_aug_back","mla_aug2"]
model_names = [
    # "mla_head1_dim256",
    "mla_aug_da_tune",
    # "mla_head2_dim512",
]
checkpoint_path = os.path.join("data", "checkpoints")
eval_folder = "evals"
s = "seen"
val_set = "val_%s" % (s)
summary_file = "summary_%s_base.json" % (s)
summary = {}
metrics = []
for model_name in model_names:
    print(model_name)
    result = {}
    result_files = list(
        os.listdir(os.path.join(checkpoint_path, model_name, eval_folder))
    )
    for result_file in result_files:
        print(result_file)
        if val_set in result_file and "json" in result_file:
            path = os.path.join(checkpoint_path, model_name, eval_folder, result_file)
            with open(path, "r") as f:
                data = json.loads(f.read())
            idx = int(result_file.split("_")[2])
            result[idx] = data
    result = sorted(result.items(), key=lambda d: d[0], reverse=False)
    index = [int(v[0]) for v in result]
    result = [v[1] for v in result]
    metrics = list(result[0].keys())
    result = dict(
        zip(
            list(result[0].keys()),
            list(zip(*[[k[1] for k in v.items()] for v in result])),
        )
    )
    for k, v in result.items():
        result[k] = dict(zip(index, v))
    summary[model_name] = result
with open(os.path.join(checkpoint_path, summary_file), "w") as f:
    f.write(json.dumps(summary))

with open(os.path.join(checkpoint_path, summary_file), "r") as f:
    summary = json.loads(f.read())
metrics = ["success", "spl"]

fig, ax = plt.subplots(len(metrics), 1, figsize=(12.8, 3.6 * len(metrics)))
for a in ax:
    a.set_prop_cycle(
        marker=["o", "+", "x", "*", "D"],
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        linestyle=["-", "--", "-.", ":", ":"],
    )
for i, metric in enumerate(metrics):
    # fig, ax = plt.subplots()
    for k, v in summary.items():
        now = sorted([(int(z), t) for z, t in v[metric].items()])
        now = dict(now)
        x = now.keys()
        y = now.values()
        ax[i].plot(x, y, label=k)
    ax[i].legend(fontsize=14)
    ax[i].set_title(val_set + "_" + metric, fontsize=20)
    ax[i].tick_params(labelsize=14)
fig.tight_layout()
fig.savefig(
    os.path.join("data", "plots", "%s_%d.jpg" % (val_set, time.time())),
    dpi=300,
)

# with open(os.path.join(checkpoint_path, summary_file), "r") as f:
#     summary = json.loads(f.read())
# del summary["cma_pm_aug"]
# for metric in ["success", "spl"]:
#     fig, ax = plt.subplots(2, 2)
#     for i, (k, v) in enumerate(summary.items()):
#         ax[int(i/2), int(i%2)].plot(v[metric], label=k)
#         ax[int(i/2), int(i%2)].set_title(k)
#     fig.suptitle(val_set+"_"+metric)
#     fig.tight_layout()
#     fig.savefig(os.path.join("data", "plots", "plots_success", "%s_%s.jpg"%(val_set, metric)))
