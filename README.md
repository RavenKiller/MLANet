<div align="center">

<h1>A Multilevel Attention Network with Sub-Instructions for Continuous Vision-and-Language Navigation</h1>

<div>
    <a href='https://link.springer.com/article/10.1007/s10489-025-06544-9' target='_blank'>[Paper (Applied Intelligence)]</a>
</div>
</div>


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

Our config files is stored in `mlanet/config/mla`:
| File | Meaning |
| ---- | ---- |
| `mla.yaml` | Train model |
| `mla_da.yaml` | Train model with DAgger |
| `mla_aug.yaml` | Train model with EnvDrop augmentation |
| `mla_da_aug_tune.yaml` | Fine-tune model with DAgger |
| `mla_ppo.yaml` | Fine-tune model with PPO |
| `mla_ablate.yaml` | Ablation study |
| `eval_single.yaml` | Evaluate and visualize a single path |
| `mla_real.yaml` | Real-world open-loop test |
| `mla_alkaid.yaml` | Real-world close-loop test on the Alkaid robot |


## Performance
The best model on validation sets is trained with EnvDrop augmentation and then fine-tuned with DAgger and PPO. We use the same strategy to train the model submitted to the test [leaderboard](https://eval.ai/web/challenges/challenge-page/719/leaderboard/1966), but on all available data (train, val_seen and val_unseen).

| Split      | TL   | NE   | OSR  | SR   | SPL  |
|:----------:|:----:|:----:|:----:|:----:|:----:|
| Test       | 7.42 | 6.78 | 0.39 | 0.34 | 0.32 |
| Val Unseen | 7.21 | 6.30 | 0.42 | 0.38 | 0.35 |
| Val Seen   | 8.10 | 5.83 | 0.50 | 0.44 | 0.42 |

## Qualitative examples

+ Val unseen, episode 7. Go straight past the pool. Walk between the bar and chairs.  Stop when you get to the corner of the bar.  That's where you will wait. (success)

https://github.com/user-attachments/assets/9c458100-7276-4213-8a9f-e929e5166cb9

+ Val unseen, episode 90. Walk into the dining area and make a right when you get to the end of the table. Walk down the hall and stand in front of the door of the dining room at the end of the hall. (failure)


https://github.com/user-attachments/assets/8332e1a4-6375-49f2-8fe6-be7e934c8a37


+ Val unseen, episode 1124. Walk up the stairs and go left into the bedroom. Turn left into the bathroom. (success)


https://github.com/user-attachments/assets/dce7affc-ee1d-4437-b5cb-f1f3af0ab578

+ Val unseen, episode 1584. Go left around the wooden barrier and stop once you reach the wooden barrier on the opposite corner. (failure)

https://github.com/user-attachments/assets/2e0b80bc-505d-47b4-9ff1-07c47787b881

+ Val seen, episode 12. move forward in front of the television. turn left and exit the room.  go down hallway and step into the bedroom on the left. (success)


https://github.com/user-attachments/assets/a9d51795-1952-4be9-b8f0-d608044cb16f

+ Val seen, episode 369. Leave the playroom and walk straight ahead. Walk to the balcony across from the balcony. Stop in front of the balcony. (failure)


https://github.com/user-attachments/assets/19f36430-4281-4167-9fc6-8d0871b598bb



## Real-world application

<img src="https://github.com/RavenKiller/MLANet/assets/41775391/6ec680f9-b19a-4b75-9f4d-80aaf342ce8e" alt="alkaid_robot" width="300">

Alkaid is a self-developed interactive service robot. Here are some parameters:

+ Camera: 720P resolution, 90° max FOV
+ Screen: 1080P, touch screen
+ Microphone: 4-microphone circular array, 61dB SNR
+ Speaker: 2 stereo units, 150Hz-20kHz output
+ Chassis: 2-wheel differential drive, 0.5m/s max speed, 1.2rad/s max angular speed

The model is evaluated on collected VLNCE@TJ validation set ([13 examples](https://www.jianguoyun.com/p/DcB0_TwQlY_kBxivhrsFIAA), extraction code: `evop`). Demonstrations ([click](http://starzx.top:5555/vlncetj_alkaid.mp4) to watch the full video):

[![Watch the video](https://github.com/RavenKiller/MLANet/assets/41775391/9109d985-b349-4cc2-acc7-251c121f5660)](http://starzx.top:5555/vlncetj_alkaid.mp4)



## Checkpoints
<!-- \[[val](https://www.jianguoyun.com/p/DSOE7KcQhY--CRjUtMMEIAA )\] \[[test](https://www.jianguoyun.com/p/DSYqcBcQhY--CRiAkbkEIAA)\] -->
\[[best model](https://www.jianguoyun.com/p/DSAuJqsQhY--CRjYrOwEIAA)\]

## Citation
```
@article{he2025multilevel,
  title = {A Multilevel Attention Network with Sub-Instructions for Continuous Vision-and-Language Navigation},
  author = {He, Zongtao and Wang, Liuyi and Li, Shu and Yan, Qingqing and Liu, Chengju and Chen, Qijun},
  year = {2025},
  month = apr,
  journal = {Applied Intelligence},
  volume = {55},
  number = {7},
  pages = {657},
  issn = {1573-7497},
  doi = {10.1007/s10489-025-06544-9}
}
```

