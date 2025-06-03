# bicycle_rl_control

This repository contains a reinforcement learning-based controller for bicycle balancing and navigation in simulation. The project uses the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator and implements PPO (Proximal Policy Optimization) for training bicycle control policies.

https://github.com/user-attachments/assets/939cacc8-8959-45d2-9751-0381ae742cec

## installation
```
cd bicycle_rl_control
./docker_build.sh
```

### make container
```
./docker_run.sh
```

### Training

Train the bicycle policy using the `BicycleEnv` environment.

Run with:

```bash
python scripts/position_tracking/bicycle_train.py -e bicycle-policy -B 8192 --max_iterations 3000
```

Train with visualization:

```bash
python scripts/position_tracking/bicycle_train.py -e bicycle-policy -B 8192 --max_iterations 3000 -v
```

### Evaluation

Evaluate the trained bicycle policy.

Run with:

```bash
python scripts/position_tracking/bicycle_eval.py -e bicycle-policy --ckpt 3000 --record
```

**Note**: If you experience slow performance or encounter other issues 
during evaluation, try removing the `--record` option.

## Reference
- 3d model [Autonomous bicycle](https://autonomous-bicycle.readthedocs.io/en/latest/) (Apache License 2.0)
