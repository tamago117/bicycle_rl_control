import numpy as np
import torch
import genesis as gs

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=30,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
bicycle = scene.add_entity(
    gs.morphs.URDF(
        file="model/bicycle.urdf",
    ),
)
########################## build ##########################
scene.build()

jnt_names = [
    "main_frame_wheel_rear",
    "main_frame_link_handle",
]
dofs_idx = [bicycle.get_joint(name).dof_idx_local for name in jnt_names]

############ Optional: set control gains ############
# set positional gains
bicycle.set_dofs_kp(
    kp=np.array([1000, 1000]),
    dofs_idx_local=dofs_idx,
)
# set velocity gains
bicycle.set_dofs_kv(
    kv=np.array([450, 450]),
    dofs_idx_local=dofs_idx,
)
# set force range for safety
bicycle.set_dofs_force_range(
    lower=np.array([-150, -200]),
    upper=np.array([150, 200]),
    dofs_idx_local=dofs_idx,
)

# PD control
for i in range(1250):
    # steering angle
    bicycle.control_dofs_position(
        np.array([1.0]),
        dofs_idx[1:],
    )
    # rear wheel velocity
    bicycle.control_dofs_velocity(
        np.array([20.0]),
        dofs_idx[:1],
    )
    # This is the control force computed based on the given control command
    # If using force control, it's the same as the given control command
    print("control force:", bicycle.get_dofs_control_force(dofs_idx))

    # This is the actual force experienced by the dof
    print("robot position:", bicycle.get_pos(), "robot velocity:", bicycle.get_vel())
    # quat to euler
    euler = gs.utils.geom.quat_to_xyz(
        gs.utils.geom.transform_quat_by_quat(
            torch.ones_like(bicycle.get_quat()) * bicycle.get_quat(), bicycle.get_quat()
        )
    )
    print("robot euler:", euler)
    print("robot dofs position:", bicycle.get_dofs_position(dofs_idx))

    scene.step()
