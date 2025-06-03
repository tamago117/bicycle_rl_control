import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from circuit_env.path_generate import make_csv_paths


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

def arrange_angle(angle):
    angle = torch.where(angle > 3.14159, angle - 2 * 3.14159, angle)
    angle = torch.where(angle < -3.14159, angle + 2 * 3.14159, angle)
    return angle


class BicycleEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        """
        BicycleEnv

        Args:
            num_envs (int): number of environments
            env_cfg (dict): environment configuration
            obs_cfg (dict): observation configuration
            reward_cfg (dict): reward configuration
            command_cfg (dict): command configuration
            show_viewer (bool): whether to show viewer
            device (str): device
        """
        # device
        self.device = torch.device(device)

        # environment
        self.num_envs = num_envs
        # observation
        self.num_obs = obs_cfg["num_obs"]
        # privileged observation
        self.num_privileged_obs = None
        # action
        self.num_actions = env_cfg["num_actions"]
        # command
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.dt = 0.02  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        # config
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        # scale
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=10),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())
        # self.scene.add_entity(gs.morphs.URDF(file="model/plane.urdf", fixed=True))

        # add target
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.35,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
            # 二輪車から目標点までの線を描画するためのCylinder
            self.target_line = self.scene.add_entity(
                morph=gs.morphs.Cylinder(
                    radius=0.02,
                    height=1.0,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 1.0, 0.0, 0.5),  # 黄色、半透明
                    ),
                ),
            )
        else:
            self.target = None
            self.target_line = None

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add bicycle
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.bicycle = self.scene.add_entity(
            gs.morphs.URDF(
                file="model/bicycle.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            )
        )

        # レーシングコースのパス情報を読み込み
        if "circuit_csv_path" in env_cfg and env_cfg["circuit_csv_path"]:
            self.circuit_path, self.right_path, self.left_path = make_csv_paths(
                env_cfg["circuit_csv_path"], 
                DL=env_cfg.get("path_resolution", 0.1)
            )
            # numpy array を torch tensor に変換
            self.circuit_path = torch.tensor(self.circuit_path, device=self.device, dtype=gs.tc_float)
            self.right_path = torch.tensor(self.right_path, device=self.device, dtype=gs.tc_float)
            self.left_path = torch.tensor(self.left_path, device=self.device, dtype=gs.tc_float)
            
            # パス関連の変数を初期化
            self.path_progress = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
            self.closest_path_idx = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
            self.path_completed = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
            self.use_circuit = True
        else:
            self.use_circuit = False

        # 地面を追加
        self.ground = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=[100.0, 100.0, 0.1],
                fixed=True,
                collision=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(0.5, 0.5, 0.5),
                ),
            ),
        )

        # build scene
        self.scene.build(n_envs=num_envs)

        # パスの可視化用の設定
        self.path_visualization = {
            "side_line": {  # 左右の線の共通設定
                "radius": 0.1,
                "colors": {
                    "right": (0.0, 1.0, 0.0, 1.0),  # 緑色
                    "left": (0.0, 1.0, 0.0, 1.0),   # 緑色
                }
            },
            "direction_arrow": {
                "radius": 0.3,
                "color": (0.0, 0.0, 1.0, 1.0),  # 青色
                "interval": 40,  # 矢印の間隔（パス長の1/20）
            },
            "target_point": {
                "radius": 0.2,  # 目標点の大きさ
                "color": (1.0, 0.0, 0.0, 1.0),  # 赤色
            }
        }

        # パスの可視化用のエンティティを追加
        if show_viewer and self.use_circuit:
            # 左右の線を可視化
            for side, path in [("right", self.right_path), ("left", self.left_path)]:
                for i in range(len(path) - 1):
                    start = path[i, :2].cpu().numpy()
                    end = path[i + 1, :2].cpu().numpy()
                    
                    # 2次元座標を3次元座標に変換（z=0.05で少し浮かせる）
                    start_3d = np.append(start, 0.05)
                    end_3d = np.append(end, 0.05)
                    
                    segment = self.scene.draw_debug_line(
                        start=start_3d,
                        end=end_3d,
                        radius=self.path_visualization["side_line"]["radius"],
                        color=self.path_visualization["side_line"]["colors"][side],
                    )
            
            # 進行方向を示す矢印を可視化
            arrow_interval = max(1, len(self.circuit_path) // self.path_visualization["direction_arrow"]["interval"])
            for i in range(0, len(self.circuit_path), arrow_interval):
                point = self.circuit_path[i, :2].cpu().numpy()
                angle = self.circuit_path[i, 2].cpu().numpy()
                
                # 2次元座標を3次元座標に変換（z=0.05で少し浮かせる）
                point_3d = np.append(point, 0.05)
                
                # デバッグ矢印を使用して進行方向を描画
                direction = np.array([np.cos(angle), np.sin(angle), 0.0])
                arrow = self.scene.draw_debug_arrow(
                    pos=point_3d,
                    vec=direction,
                    radius=self.path_visualization["direction_arrow"]["radius"],
                    color=self.path_visualization["direction_arrow"]["color"],
                )


        # names to indices
        self.motor_dofs = [self.bicycle.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.bicycle.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.bicycle.set_dofs_kv([self.env_cfg["kv"]] * self.num_actions, self.motor_dofs)
        # force range
        self.bicycle.set_dofs_force_range(
            lower=torch.tensor([-self.env_cfg["max_actuator_force"], -self.env_cfg["max_actuator_force"]], device=self.device),
            upper=torch.tensor([self.env_cfg["max_actuator_force"], self.env_cfg["max_actuator_force"]], device=self.device),
            dofs_idx_local=self.motor_dofs,
        )

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)

        # action
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        # state
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        """
        Resample commands for path following

        Args:
            envs_idx (torch.Tensor): indices of environments
        """
        if self.use_circuit:
            # レーシングコースの場合はパス上の次の目標点を設定
            self._update_path_targets(envs_idx)
            
            # 目標点を可視化
            if self.target is not None:
                target_pos = self.commands[envs_idx, :2].cpu().numpy()  # 最初の環境の目標点
                # 2次元テンソルとして位置を設定
                target_pos_3d = np.zeros((len(envs_idx), 3))
                target_pos_3d[:, :2] = target_pos
                target_pos_3d[:, 2] = 0.05  # z座標を追加
                # print(target_pos_3d)
                self.target.set_pos(target_pos_3d, zero_velocity=True, envs_idx=envs_idx)

    def _update_path_targets(self, envs_idx):
        """
        パス上の次の目標点を更新
        
        Args:
            envs_idx (torch.Tensor): 更新する環境のインデックス
        """
        for env_idx in envs_idx:
            # 現在位置から最も近いパス点を見つける
            current_pos = self.base_pos[env_idx, :2]  # x, y のみ
            distances = torch.norm(self.circuit_path[:, :2] - current_pos, dim=1)
            closest_idx = torch.argmin(distances)
            self.closest_path_idx[env_idx] = closest_idx
            
            # 進行方向の目標点を設定（少し先の点を目標とする）
            lookahead_distance = self.env_cfg.get("lookahead_steps", 10)
            target_idx = (closest_idx + lookahead_distance) % len(self.circuit_path)
            
            # 目標位置を設定
            self.commands[env_idx, :3] = self.circuit_path[target_idx, :3]

    def _check_path_progress(self):
        """
        パス上の進行状況をチェック
        """
        if not self.use_circuit:
            return torch.tensor([], device=self.device, dtype=torch.long)
            
        # パス完了をチェック（一周完了）
        progress_threshold = self.env_cfg.get("progress_threshold", 0.8)
        completed_envs = []
        
        for env_idx in range(self.num_envs):
            current_progress = self.closest_path_idx[env_idx].float() / len(self.circuit_path)
            if current_progress > progress_threshold and not self.path_completed[env_idx]:
                self.path_completed[env_idx] = True
                completed_envs.append(env_idx)
                
        return torch.tensor(completed_envs, device=self.device, dtype=torch.long)

    def _get_path_deviation(self):
        """
        パスからの距離を計算
        
        Returns:
            path_deviation (torch.Tensor): パスからの横方向距離
            progress_error (torch.Tensor): パス進行方向の誤差
        """
        if not self.use_circuit:
            return torch.zeros((self.num_envs,), device=self.device), torch.zeros((self.num_envs,), device=self.device)
            
        path_deviation = torch.zeros((self.num_envs,), device=self.device)
        progress_error = torch.zeros((self.num_envs,), device=self.device)
        
        for env_idx in range(self.num_envs):
            current_pos = self.base_pos[env_idx, :2]
            closest_idx = self.closest_path_idx[env_idx]
            
            # 最も近いパス点
            closest_point = self.circuit_path[closest_idx, :2]
            path_angle = self.circuit_path[closest_idx, 2]
            
            # パスからの距離ベクトル
            deviation_vec = current_pos - closest_point
            
            # パス方向と垂直方向の成分を計算
            path_direction = torch.tensor([torch.cos(path_angle), torch.sin(path_angle)], device=self.device)
            path_normal = torch.tensor([-torch.sin(path_angle), torch.cos(path_angle)], device=self.device)
            
            # 横方向距離（パスからの横ずれ）
            path_deviation[env_idx] = torch.abs(torch.dot(deviation_vec, path_normal))
            
            # 進行方向誤差
            progress_error[env_idx] = torch.dot(deviation_vec, path_direction)
            
        return path_deviation, progress_error

    def _update_target_line(self):
        """二輪車から目標点までの線を更新"""
        if self.target_line is not None:
            # 二輪車の位置
            bicycle_pos = self.base_pos[0].cpu().numpy()  # 最初の環境の二輪車の位置
            # 目標点の位置
            target_pos = self.commands[0, :2].cpu().numpy()  # 最初の環境の目標点
            target_pos_3d = np.append(target_pos, 0.05)  # z座標を追加

            # 二点間の距離を計算
            distance = np.linalg.norm(target_pos_3d - bicycle_pos)
            
            # 二点間の方向ベクトルを計算
            direction = target_pos_3d - bicycle_pos
            direction = direction / np.linalg.norm(direction)
            
            # 回転行列を計算（z軸からdirectionへの回転）
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, direction)
            if np.linalg.norm(rotation_axis) > 0:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                rotation_angle = np.arccos(np.dot(z_axis, direction))
                # 回転軸と角度をnumpy配列として渡す（形状を修正）
                rotation_matrix = gs.utils.geom.axis_angle_to_quat(
                    np.array([rotation_axis], dtype=np.float32),  # 2次元配列として渡す
                    np.array([rotation_angle], dtype=np.float32)  # 2次元配列として渡す
                )
                # クォータニオンを2次元テンソルとして渡す
                quat_2d = np.array([[rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], rotation_matrix[0, 3]]], dtype=np.float32)
            else:
                # 回転がない場合は単位クォータニオンを使用
                quat_2d = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

            # 線の位置と向きを更新
            self.target_line.set_pos(np.array([bicycle_pos], dtype=np.float32), zero_velocity=True)  # 2次元配列として渡す
            self.target_line.set_quat(quat_2d, zero_velocity=True)
            # 線の長さを更新
            self.target_line.set_scale([1.0, 1.0, distance])

    def step(self, actions):
        """
        Step the environment

        Args:
            actions (torch.Tensor): actions

        Returns:
            obs (torch.Tensor): observations
            rew (torch.Tensor): rewards
            reset (torch.Tensor): reset flags
            extras (dict): extra information
        """
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        #  rear wheel action range: 0 ~ 1
        self.actions[:, :1] = torch.clip(self.actions[:, :1], 0, self.env_cfg["clip_actions"])
        # consider action latency
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        # input 0: rear wheel velocity, 1: steer angle
        rear_wheel_velocity = (exec_actions[:, :1] * self.env_cfg["max_rear_wheel_velocity"]).clone().detach()
        steer_angle = (exec_actions[:, 1:] * self.env_cfg["max_steer_angle"]).clone().detach()

        # calculate rear wheel velocity and steer angle from input
        self.bicycle.control_dofs_velocity(rear_wheel_velocity.cpu().numpy(), self.motor_dofs[:1])
        self.bicycle.control_dofs_position(steer_angle.cpu().numpy(), self.motor_dofs[1:])
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.bicycle.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.bicycle.get_quat()
        # [pitch, roll, yaw] (degree)
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.bicycle.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.bicycle.get_ang(), inv_base_quat)
        self.dof_pos[:] = self.bicycle.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.bicycle.get_dofs_velocity(self.motor_dofs)

        # パス関連の更新
        if self.use_circuit:
            # すべての環境に対して目標点を更新
            self._resample_commands(torch.arange(self.num_envs, device=self.device))
            # 二輪車から目標点までの線を更新
            self._update_target_line()
            
        # resample commands
        envs_idx = self._check_path_progress()
        finish_idx = envs_idx

        # check termination and reset
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # finish reward
        self.rew_buf[finish_idx] += 100
        # self.episode_sums["finish"] += 10

        # パス情報を観測に追加
        if self.use_circuit:
            path_deviation, progress_error = self._get_path_deviation()
            
            # 観測の構成を更新
            self.obs_buf = torch.cat(
                [
                    torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1), # 3
                    self.base_quat, # 4
                    torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1), # 3
                    torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1), # 3
                    self.dof_pos[:,1:], # 1
                    self.dof_vel, # 2
                    self.last_actions, # 2
                    torch.clip(path_deviation.unsqueeze(1) * self.obs_scales.get("path_deviation", 1.0), -1, 1), # 1
                    torch.clip(progress_error.unsqueeze(1) * self.obs_scales.get("progress_error", 1.0), -1, 1), # 1
                ],
                axis=-1,
            )
        else:
            # 既存の観測構成
            self.obs_buf = torch.cat(
                [
                    torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1), # 3
                    self.base_quat, # 4
                    torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1), # 3
                    torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1), # 3
                    self.dof_pos[:,1:], # 1
                    self.dof_vel, # 2
                    self.last_actions, # 2
                ],
                axis=-1,
            )

        self.last_actions[:] = self.actions[:]
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        """
        Get observations

        Returns:
            obs (torch.Tensor): observations
        """
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        """
        Get privileged observations

        Returns:
            obs (torch.Tensor): privileged observations
        """
        return None

    def reset_idx(self, envs_idx):
        """
        Reset indices

        Args:
            envs_idx (torch.Tensor): indices of environments
        """
        if len(envs_idx) == 0:
            return
        
        # reset dofs 0
        self.dof_pos[envs_idx] = 0.0
        self.dof_vel[envs_idx] = 0.0
        self.bicycle.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.bicycle.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.bicycle.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.bicycle.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

        # パス関連の状態をリセット
        if self.use_circuit:
            self.path_progress[envs_idx] = 0
            self.closest_path_idx[envs_idx] = 0
            self.path_completed[envs_idx] = False
            
            # スタート地点をパスの最初の点に設定
            start_idx = torch.randint(0, min(len(self.circuit_path), 10), (len(envs_idx),), device=self.device)
            for i, env_idx in enumerate(envs_idx):
                start_pos = self.circuit_path[start_idx[i], :2]
                start_angle = self.circuit_path[start_idx[i], 2]
                
                self.base_pos[env_idx, :2] = start_pos
                self.last_base_pos[env_idx, :2] = start_pos
                
                # パスの角度に合わせて初期姿勢を設定
                quat_z = torch.cos(start_angle / 2)
                quat_w = torch.sin(start_angle / 2)
                self.base_quat[env_idx] = torch.tensor([quat_w, 0, 0, quat_z], device=self.device)
                
                # 2次元テンソルとして位置を設定
                pos_2d = self.base_pos[env_idx].unsqueeze(0)
                quat_2d = self.base_quat[env_idx].unsqueeze(0)
                self.bicycle.set_pos(pos_2d, zero_velocity=True, envs_idx=torch.tensor([env_idx], device=self.device))
                self.bicycle.set_quat(quat_2d, zero_velocity=True, envs_idx=torch.tensor([env_idx], device=self.device))

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions for path following ----------------
    def _reward_path_following(self):
        """パス追従報酬"""
        if not self.use_circuit:
            return torch.zeros((self.num_envs,), device=self.device)
            
        path_deviation, progress_error = self._get_path_deviation()
        
        # パスからの距離に基づく報酬（近いほど高い報酬）
        deviation_reward = torch.exp(-path_deviation * self.reward_cfg.get("path_deviation_scale", 2.0))
        
        return deviation_reward
        
    def _reward_path_progress(self):
        """パス進行報酬"""
        if not self.use_circuit:
            return torch.zeros((self.num_envs,), device=self.device)
            
        # 前進した距離に基づく報酬
        progress_reward = torch.zeros((self.num_envs,), device=self.device)
        
        for env_idx in range(self.num_envs):
            current_idx = self.closest_path_idx[env_idx]
            if hasattr(self, 'last_closest_path_idx'):
                last_idx = self.last_closest_path_idx[env_idx]
                
                # パス上での進行を計算
                progress_diff = (current_idx - last_idx) % len(self.circuit_path)
                if progress_diff < len(self.circuit_path) // 2:  # 逆走を避ける
                    progress_reward[env_idx] = progress_diff.float()
                    
        # 前のインデックスを保存
        if not hasattr(self, 'last_closest_path_idx'):
            self.last_closest_path_idx = torch.zeros_like(self.closest_path_idx)
        self.last_closest_path_idx[:] = self.closest_path_idx[:]
        
        return progress_reward

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_roll(self):
        roll = self.base_euler[:, 1]
        roll = torch.where(roll > 180, roll - 360, roll) / 180 * 3.14159  # use rad for roll_reward
        roll_rew = torch.exp(self.reward_cfg["roll_lambda"] * torch.abs(roll))
        # roll_rew = torch.abs(roll/3.14159)
        return roll_rew

    def _reward_angular_yaw(self):
        angular_yaw = self.base_ang_vel[:, 2]
        angular_rew = torch.abs(angular_yaw/3.14159)
        return angular_rew

    def _reward_angular_roll(self):
        angular_roll = self.base_ang_vel[:, 1]
        angular_rew = torch.abs(angular_roll/3.14159)
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew
