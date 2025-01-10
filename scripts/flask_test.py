import requests
from base64 import b64encode
import os
import json
import time

# res = requests.post("http://localhost:32145/reset_hiddens")
# print(res.text)
SERVER_STR = "106.14.157.21:5588"
os.system("cp -r data/vlntj-ce-extend-val data/vlntj-ce-extend-val-new")
time_infos = []
for ep_id in range(13):
    res = requests.post("http://{}/reset_hiddens".format(SERVER_STR))
    steps = sorted(
        [
            int(v.split(".")[0])
            for v in os.listdir("data/vlntj-ce-extend-val/{}/rgb".format(ep_id))
        ]
    )
    pred_action_close = []
    for step_id in steps:
        tic = time.time()
        with open(
            "data/vlntj-ce-extend-val/{}/rgb/{}.png".format(ep_id, step_id),
            "rb",
        ) as f:
            jpg_data = b64encode(f.read())
        with open(
            "data/vlntj-ce-extend-val/{}/depth/{}.png".format(ep_id, step_id),
            "rb",
        ) as f:
            depth_data = b64encode(f.read())
        # print("==========")
        time_info_encode = time.time() - tic
        tic = time.time()
        res = requests.post(
            "http://{}/predict_action".format(SERVER_STR),
            data={"ep_id": ep_id, "rgb": jpg_data},
        )
        # print(ep_id, step_id)
        time_info_prediction = time.time() - tic

        time_info = json.loads(res.text)["time_info"]
        time_info["encode_b64"] = time_info_encode
        time_info["network_latency"] = (
            time_info_prediction
            - time_info["decode_b64"]
            - time_info["forward"]
            - time_info["observation"]
        )
        time_info["total"] = time_info_prediction + time_info_encode
        time_infos.append(time_info)
        pred_action_close.append(json.loads(res.text)["action"])
    os.makedirs(
        "data/vlntj-ce-extend-val-new/{}/pred_action_close".format(ep_id),
        exist_ok=True,
    )
    with open(
        "data/vlntj-ce-extend-val-new/{}/pred_action_close/pred_action_close.json".format(
            ep_id
        ),
        "w",
    ) as f:
        f.write(json.dumps(pred_action_close))
res = requests.post("http://{}/reset_hiddens".format(SERVER_STR))
with open("time_infos.json", "w") as f:
    f.write(json.dumps(time_infos, indent=2))

import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.append("/share/home/tj90055/hzt/MLANet")
from vlnce_baselines.common.utils import Metrics

DATA_FOLDER = Path("data/vlntj-ce-extend-val-new")
results = {
    "sr": [],
    "spl": [],
    "apa": [],
    "ndtw": [],
}
for i in range(13):
    episode_folder = DATA_FOLDER / str(i)
    with open(episode_folder / "action" / "action.json", "r") as f:
        gt_action = json.loads(f.read())
    with open(
        episode_folder / "pred_action_close" / "pred_action_close.json", "r"
    ) as f:
        pred_action = json.loads(f.read())
    metrics = Metrics(pred_action, gt_action)
    results["sr"].append(metrics.calc_sr())
    results["spl"].append(metrics.calc_spl())
    results["apa"].append(metrics.calc_apa())
    results["ndtw"].append(metrics.calc_ndtw())
    metrics.plot_pos(DATA_FOLDER, i, show=False)
print(results)
mean_results = {k: np.mean(v) for k, v in results.items()}
print(mean_results)
