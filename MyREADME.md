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
# pufferdrive
python waymax_interface/run_waymax.py -f 0 -s 3 --video
# log replay
python waymax_interface/run_waymax.py -f 0 -s 3 --video --policy expert

```


### PufferDrive Eval

human-replay, sdc-only

```bash

puffer eval puffer_drive --load-model-path ./experiments/puffer_drive_177878959462.pt --eval.map-dir ./resources/drive/binaries/training/ --eval.sample-mode sequential --env.num-maps 5 --env.num-agents 64 --env.control-mode control_sdc_only
```