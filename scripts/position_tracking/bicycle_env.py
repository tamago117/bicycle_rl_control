import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


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
                    scale=0.25,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
        else:
            self.target = None

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

        # build scene
        self.scene.build(n_envs=num_envs)

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
        Resample position(2D) commands for the bicycle

        Args:
            envs_idx (torch.Tensor): indices of environments
        """
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), self.device)
        # z is 0(device)
        self.commands[envs_idx, 2] = torch.zeros((len(envs_idx),), device=self.device)
        if self.target is not None:
            self.target.set_pos(self.commands[envs_idx], zero_velocity=True, envs_idx=envs_idx)

    def _at_target(self):
        """
        Check if the bicycle is at the target

        Returns:
            at_target (torch.Tensor): indices of environments at the target
        """
        at_target = (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"]).nonzero(as_tuple=False).flatten()
        )
        return at_target

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

        # resample commands
        envs_idx = self._at_target()
        finish_idx = envs_idx
        self._resample_commands(envs_idx)

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

        # compute observations
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1), # 3
                self.base_quat, # 4
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1), # 3
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1), # 3
                self.dof_pos[:,1:], # 1
                self.dof_vel, # 2
                self.last_actions, # 2 sum : 19
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

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        return target_rew
    
    def _reward_target_angle(self):
        target_angle = torch.atan2(self.rel_pos[:, 1], self.rel_pos[:, 0])
        current_angle = arrange_angle((self.base_euler[:, 1] + 90) * 3.14159 / 180)
        angle_diff = torch.abs(arrange_angle(target_angle - current_angle))/3.14159
        return angle_diff

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
