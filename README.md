# Multi-Level Attention with Sub-instruction for Continuous Vision-and-Language Navigation
<!-- Official implrementations of *Multi-Level Attention with Sub-instruction for
Continuous Vision-and-Language Navigation* ([paper]()) -->
*This repository will be completed after being reviewed.*


## Setup
1. Use [anaconda](https://anaconda.org/) to create a Python 3.6 environment:
```bash
conda create -n vlnce python3.6
conda activate vlnce
```
2. Install [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7) 0.1.7:
```bash
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless
```
3. Install [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7) 0.1.7:
```bash
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
# installs both habitat and habitat_baselines
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```
*Habitat v0.2.1 is also supported now!*
4. Clone this repository and install python requirements:
```bash
git clone https://github.com/RavenKiller/MLA.git
cd MLA
pip install -r requirements.txt
```
5. Download Matterport3D sences:
   + Get the official `download_mp.py` from [Matterport3D project webpage](https://niessner.github.io/Matterport/)
   + Download scene data for Habitat
    ```bash
    # requires running with python 2.7
    python download_mp.py --task habitat -o data/scene_datasets/mp3d/
    ```
   + Extract such that it has the form `data/scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 scenes.
6. Download preprocessed episodes [R2R_VLNCE_FSASub](https://github.com/RavenKiller/R2R_VLNCE_FSASub) from [here](https://drive.google.com/file/d/1rJn2cvhlQ7-GZ-gcUjJAjbyxfguiz2vv/view?usp=sharing). Extrach it into `data/datasets/`.
7. Download the depth encoder `gibson-2plus-resnet50.pth` from [here](https://github.com/facebookresearch/habitat-lab/tree/master/habitat_baselines/rl/ddppo). Extract the contents to `data/ddppo-models/{model}.pth`.

## Train, evaluate and test
`run.py` is the program entrance. You can run it like:
```bash
python run.py \
  --exp-config {config} \
  --run-type {type}
```
`{config}` should be replaced by a config file path; `{type}` should be `train`, `eval` or `inference`, meaning train models, evaluate models and test models.

Our config files is stored in `vlnce_baselines/config/mla`:
| File | Meaning |
| ---- | ---- |
| `mla.yaml` | Train model |
| `mla_da.yaml` | Train model with DAgger |
| `mla_aug.yaml` | Train model with EnvDrop augmentation |
| `mla_da_aug_tune.yaml` | Fine-tune model with DAgger |
| `mla_ppo.yaml` | Fine-tune model with PPO |
| `mla_ablate.yaml` | Ablation study |
| `eval_single.yaml` | Evaluate and visualize a single path |




## Performance
The best model on validation sets is trained with EnvDrop augmentation and then fine-tuned with DAgger and PPO. We use the same strategy to train the model submitted to the test [leaderboard](https://eval.ai/web/challenges/challenge-page/719/leaderboard/1966), but on all available data (train, val_seen and val_unseen).

| Split      | TL   | NE   | OSR  | SR   | SPL  |
|:----------:|:----:|:----:|:----:|:----:|:----:|
| Test       | 7.42 | 6.78 | 0.39 | 0.34 | 0.32 |
| Val Unseen | 7.21 | 6.30 | 0.42 | 0.38 | 0.35 |
| Val Seen   | 8.10 | 5.83 | 0.50 | 0.44 | 0.42 |

## Checkpoints
<!-- \[[val](https://www.jianguoyun.com/p/DSOE7KcQhY--CRjUtMMEIAA )\] \[[test](https://www.jianguoyun.com/p/DSYqcBcQhY--CRiAkbkEIAA)\] -->
\[[best model](https://www.jianguoyun.com/p/DSAuJqsQhY--CRjYrOwEIAA)\]

## References
+ [VLN-CE](https://github.com/jacobkrantz/VLN-CE).
