import torch

from bicycle_env import BicycleEnv

import genesis as gs


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
        "circuit_csv_path": "scripts/path_tracking/circuit_env/circuit.csv",  # CSVファイルのパス
        "path_resolution": 0.5,  # パスの解像度
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
    gs.init(logging_level="error")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    env_cfg["visualize_target"] = True

    env = BicycleEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # envの動作確認を行う
    env.reset()
    for i in range(1000):
        action = torch.zeros(1, 2, device=env.device)
        action[0, 0] = 0.0
        action[0, 1] = 0.0
        env.step(action)


if __name__ == "__main__":
    main()
