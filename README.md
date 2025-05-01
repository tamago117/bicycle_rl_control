# bicycle_rl_control

This repository contains a reinforcement learning-based controller for bicycle balancing and navigation in simulation. The project uses the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator and implements PPO (Proximal Policy Optimization) for training bicycle control policies.



### Installation

At this stage, we have defined the environments. Now, we use the PPO implementation from `rsl-rl` to train the policy. Follow these installation steps:

```bash
# Install rsl_rl.
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v1.0.2 && pip install -e .

# Install tensorboard.
pip install tensorboard
```

### Training

Train the bicycle policy using the `BicycleEnv` environment.

Run with:

```bash
python scripts/bicycle_train.py -e bicycle-policy -B 8192 --max_iterations 300
```

Train with visualization:

```bash
python scripts/bicycle_train.py -e bicycle-policy -B 8192 --max_iterations 300 -v
```

### Evaluation

Evaluate the trained bicycle policy.

Run with:

```bash
python scripts/bicycle_eval.py -e bicycle-policy --ckpt 300 --record
```

**Note**: If you experience slow performance or encounter other issues 
during evaluation, try removing the `--record` option.

## Reference
- 3d model [Autonomous bicycle](https://autonomous-bicycle.readthedocs.io/en/latest/) (Apache License 2.0)