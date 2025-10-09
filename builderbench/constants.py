import mujoco
import numpy as np

from dataclasses import dataclass
from typing import Dict, Tuple

_ARM_JOINTS = [
    'arm_tx', 
    'arm_ty', 
    'arm_tz', 
    'arm_yaw',
]

_HAND_JOINTS = [
    'left_coupler_joint', 
    'left_driver_joint', 
    'left_follower_joint', 
    'left_spring_link_joint', 
    'right_coupler_joint', 
    'right_driver_joint', 
    'right_follower_joint', 
    'right_spring_link_joint'
]

@dataclass(frozen=True)
class Dof:
    """Forearm degree of freedom."""
    joint_type: int
    axis: Tuple[int, int, int]
    stiffness: float
    range: Tuple[float, float]
    transmission: int
    
_FOREARM_DOFS: Dict[str, Dof] = {
    "arm_tx": Dof(
        joint_type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=(0, 1, 0),
        stiffness=300,
        range=(-0.05, 0.45),
        transmission=mujoco.mjtTrn.mjTRN_JOINT,
    ),
    "arm_ty": Dof(
        joint_type=mujoco.mjtJoint.mjJNT_SLIDE, 
        axis=(1, 0, 0),
        stiffness=300, 
        range=(-0.35, 0.35),
        transmission=mujoco.mjtTrn.mjTRN_JOINT,
    ),
    "arm_tz": Dof(
        joint_type=mujoco.mjtJoint.mjJNT_SLIDE, 
        axis=(0, 0, -1),
        stiffness=300, 
        range=(0.07, 0.5),
        transmission=mujoco.mjtTrn.mjTRN_JOINT,
    ),
    "arm_yaw": Dof(
        joint_type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=(0, 0, -1),
        stiffness=300, 
        range=(-1.5707, 1.5707),
        transmission=mujoco.mjtTrn.mjTRN_JOINT,
    ),
}

_CUSTOM_COLORS = [
    np.array([1.0, 0.49, 0.43, 1.0]),   # coral
    np.array([0.0, 0.58, 0.55, 1.0]),   # teal
    np.array([0.93, 0.78, 0.28, 1.0]),  # mustard
    np.array([0.38, 0.31, 0.86, 1.0]),  # indigo
    np.array([0.98, 0.67, 0.78, 1.0]),  # rose
    np.array([0.13, 0.55, 0.13, 1.0]),  # forest green
    np.array([0.85, 0.37, 0.00, 1.0]),  # burnt orange
    np.array([0.29, 0.63, 0.92, 1.0]),  # sky blue
    np.array([0.56, 0.27, 0.68, 1.0]),  # plum
    np.array([0.44, 0.50, 0.56, 1.0]),  # slate gray
]

_TIME_STEPS = {
    # env_name : [sim_dt, ctrl_dt]

    'cube-1-play' : [0.002, 0.02],
    'cube-2-play' : [0.002, 0.02],
    'cube-3-play' : [0.002, 0.02],
    'cube-4-play' : [0.002, 0.02],
    'cube-5-play' : [0.002, 0.02],
    'cube-6-play' : [0.002, 0.02],
    'cube-7-play' : [0.002, 0.02],
    'cube-8-play' : [0.002, 0.02],
    'cube-9-play' : [0.002, 0.02],

    'cube-1-task1': [0.005, 0.02],
    'cube-1-task2': [0.005, 0.02],
    'cube-1-task3': [0.005, 0.02],
    'cube-1-task4': [0.005, 0.02],
    'cube-1-task5': [0.005, 0.02],
    'cube-2-task1': [0.005, 0.02],
    'cube-2-task2': [0.005, 0.02],
    'cube-2-task3': [0.005, 0.02],
    'cube-2-task4': [0.005, 0.02],
    'cube-2-task5': [0.005, 0.02],
    'cube-3-task1': [0.005, 0.02],
    'cube-3-task2': [0.005, 0.02],
    'cube-3-task3': [0.005, 0.02],
    'cube-3-task4': [0.005, 0.02],
    'cube-3-task5': [0.005, 0.02],
    'cube-4-task1': [0.005, 0.02],
    'cube-4-task2': [0.005, 0.02],
    'cube-4-task3': [0.005, 0.02],
    'cube-4-task4': [0.005, 0.02],
    'cube-4-task5': [0.005, 0.02],
    'cube-5-task1': [0.002, 0.02],
    'cube-5-task2': [0.002, 0.02],
    'cube-5-task3': [0.002, 0.02],
    'cube-5-task4': [0.002, 0.02],
    'cube-5-task5': [0.002, 0.02],
    'cube-6-task1': [0.002, 0.02],
    'cube-6-task2': [0.002, 0.02],
    'cube-6-task3': [0.002, 0.02],
    'cube-6-task4': [0.002, 0.02],
    'cube-6-task5': [0.002, 0.02],
    'cube-7-task1': [0.002, 0.02],
    'cube-7-task2': [0.002, 0.02],
    'cube-7-task3': [0.002, 0.02],
    'cube-7-task4': [0.002, 0.02],
    'cube-7-task5': [0.002, 0.02],
    'cube-8-task1': [0.002, 0.02],
    'cube-8-task2': [0.002, 0.02],
    'cube-8-task3': [0.002, 0.02],
    'cube-8-task4': [0.002, 0.02],
    'cube-8-task5': [0.002, 0.02],
    'cube-9-task1': [0.002, 0.02],
    'cube-9-task2': [0.002, 0.02],
    'cube-9-task3': [0.002, 0.02],
    'cube-9-task4': [0.002, 0.02],
    'cube-9-task5': [0.002, 0.02],
}