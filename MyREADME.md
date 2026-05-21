## Training

### training bad_driver in pacific

```bash
puffer train bad_driver
```

### how to use carla for testing

使用carla跑scenario
```bash
# for 60 seconds
/home/tjhu78u/miniconda3/envs/py3.8torch2.4.1/bin/python carla_interface/run.py   --map Town01   --num-agents 8   --max-episode-steps 600
```


使用run.py调用puffer eval模式，输出mp4视频
```bash
/home/tjhu78u/miniconda3/envs/py3.8torch2.4.1/bin/python carla_interface/run.py   --mode puffer-eval   --map Town01   --num-agents 8   --max-episode-steps 91   --output-mp4 carla_interface/cache/Town01/test_puffer_eval.mp4
```

### how to use waymax for testing

```bash
uv pip install "numpy==2.4.5"
```

```bash
# pufferdrive
python waymax_interface/run_waymax.py -f 0 -s 3 --video

# use json format instead of raw waymo
python waymax_interface/run_waymax.py --json ./data/processed/training/tfrecord-00000-of-01000_262.json --video

# use json format for argoverse2
python waymax_interface/run_waymax.py --json ./data/processed/av2/training
# log replay
python waymax_interface/run_waymax.py -f 0 -s 3 --video --policy expert

```

TODO: waymax for evaluation 


### PufferDrive Eval

```bash
uv pip install "numpy<2.0"
```

sample video

```bash
puffer eval puffer_drive --load-model-path ./experiments/puffer_drive_177885682784.pt --eval.map-dir ./resources/drive/binaries/training/ --eval.sample-mode sequential --env.num-maps 10000 --env.num-agents 1 --env.control-mode control_sdc_only --eval.deterministic True
```

human-replay, sdc-only

```bash
# random 10000 sdc on training set, total map are 10000
puffer eval puffer_drive --load-model-path ./experiments/puffer_drive_177885682784.pt --eval.map-dir ./resources/drive/binaries/training/ --eval.sample-mode sequential --eval.human-replay-eval True --env.num-maps 10000 --eval.num-eval-agents 10000 --env.control-mode control_sdc_only --eval.deterministic True

# random 10000 sdc on validation set, total map are 10000
puffer eval puffer_drive --load-model-path ./experiments/puffer_drive_177885682784.pt --eval.map-dir ./resources/drive/binaries/validation/ --eval.sample-mode sequential --eval.human-replay-eval True --env.num-maps 10000 --eval.num-eval-agents 10000 --env.control-mode control_sdc_only --eval.deterministic True
```


use data from scenario-max

```bash
python pufferlib/ocean/drive/drive.py <path_to_json_dir>

# waymo scenario-max, video
puffer eval puffer_drive --load-model-path ./experiments/puffer_drive_177885682784.pt --eval.map-dir pufferlib/resources/drive/binaries/waymo_smax --eval.sample-mode sequential --env.num-maps 100 --env.control-mode control_sdc_only --eval.deterministic True --env.num-agents 1

# argoverse2 scenario-max, video
puffer eval puffer_drive --load-model-path ./experiments/puffer_drive_177885682784.pt --eval.map-dir pufferlib/resources/drive/binaries/av2_smax --eval.sample-mode sequential --env.num-maps 100 --env.control-mode control_sdc_only --eval.deterministic True --env.num-agents 1 

# argoverse2 scenario-max, metrics
puffer eval puffer_drive --load-model-path ./experiments/puffer_drive_177885682784.pt --eval.map-dir pufferlib/resources/drive/binaries/av2_smax --eval.sample-mode sequential --env.num-maps 100 --env.control-mode control_sdc_only --eval.deterministic True --eval.num-eval-agents 100 --eval.human-replay-eval True

# no need for this, as scenario-max already generate 91 frames format
# --env.episode-length 110 --env.resample-frequency 1100

```