import argparse
import os
import pickle
import shutil

from bicycle_env import BicycleEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 2,
        "dof_names": ["main_frame_wheel_rear", "main_frame_link_handle"],
        # termination
        "termination_if_roll_greater_than": 50,  # degree
        "termination_if_x_greater_than": 30.0,
        "termination_if_y_greater_than": 30.0,
        # dof position gain
        "kp": 1000.0,
        # dof velocity gain
        "kv": 400.0,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "at_target_threshold": 2.0,
        "resampling_time_s": 5.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
        "max_rear_wheel_velocity": 30.0, # rad/s
        "max_steer_angle": 1.0, # rad
        "max_actuator_force": 100.0, # N
        # visualization
        "visualize_target": True,
        "visualize_camera": True,
        "max_visualize_FPS": 60,
        
        # レーシングコース設定
        "circuit_csv_path": "circuit_env/circuit.csv",  # CSVファイルのパス
        "path_resolution": 0.1,  # パスの解像度
        "lookahead_steps": 10,  # 目標点の先読み距離
        "progress_threshold": 0.8,  # パス完了判定の閾値
    }
    
    obs_cfg = {
        "num_obs": 18,  # 既存の観測 + パス関連の観測
        "obs_scales": {
            "rel_pos": 1 / 10.0,
            "lin_vel": 1 / 10.0,
            "ang_vel": 1 / 3.14159,
            "path_deviation": 1.0,  # パスからの距離のスケール
            "progress_error": 1.0,  # 進行方向誤差のスケール
        },
    }
    
    reward_cfg = {
        "roll_lambda": -10.0,
        "reward_scales": {
            "target": 10.0,
            "target_angle": -0.0,
            "smooth": -1e-5,
            "roll": 0.0,
            "angular_yaw": -0.01,
            "angular_roll": -0.01,
            "crash": -5000.0,
            # パス追従関連の報酬スケール
            "path_following": 1.0,  # パス追従報酬
            "path_progress": 0.5,   # パス進行報酬
        },
        "path_deviation_scale": 2.0,  # パスからの距離の報酬スケール
    }
    
    command_cfg = {
        "num_commands": 3,  # x, y, angle
        "pos_x_range": [-15.0, 15.0],
        "pos_y_range": [-15.0, 15.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--checkpoint", type=int, default=-1)
    args = parser.parse_args()

    gs.init(logging_level="error")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    
    # 学習を再開する場合の設定
    if args.resume:
        train_cfg["runner"]["resume"] = True
        train_cfg["runner"]["checkpoint"] = args.checkpoint

    if not args.resume and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True

    env = BicycleEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )


    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    train_cfg["algorithm"]["class_name"] = "PPO"
    train_cfg["policy"]["class_name"] = "ActorCritic"

    if args.resume:
        if args.checkpoint == -1:
            #ファイル名から一番大きな数字を取得
            checkpoint_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
            checkpoint_numbers = [int(f.split("_")[1].split(".")[0]) for f in checkpoint_files]
            resume_path = os.path.join(log_dir, f"model_{max(checkpoint_numbers)}.pt")
        else:
            resume_path = os.path.join(log_dir, f"model_{args.checkpoint}.pt")
        runner.load(resume_path)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python scripts/bicycle_train.py
"""
