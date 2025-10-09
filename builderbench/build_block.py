import jax
import optax
import jax.numpy as jnp

import math
import numpy as np
import copy

import mujoco
from mujoco import rollout
from mujoco.mjx._src import math as mjx_math

from etils import epath
from ml_collections import config_dict
from typing import Optional

from builderbench.env_utils import get_assets, State
from builderbench.constants import _ARM_JOINTS, _FOREARM_DOFS, _CUSTOM_COLORS

def default_config() -> config_dict.ConfigDict:
    """Returns the default config for the creative double cube task"""
    config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        
        num_cubes=3,
        action_scale = config_dict.create(
            xyz_scale = 0.05,
            yaw_scale = 0.05,
            finger_scale = 255.0,
            ),
        episode_length=250,
        task_id=0,
        reward_sensitivity=5.0,
        success_threshold=0.02,
        easy_success_threshold=0.05,
        early_termination=True,
        permutation_invariant_reward=True,
    )
    return config

class CreativeCube():
    def __init__(
        self,
        num_envs: int,
        num_threads: int,
        config: config_dict.ConfigDict = default_config(),
    ):
        
        xml_path = (
            epath.Path(__file__).resolve().parent
            / "xmls"
            / "scene.xml"
        )
                        
        self._init(xml_path=xml_path, config=config)

        # making model and data copies for rollout 
        self.model_list = [self.model] * num_envs
        self.data_list = []
        tmp = mujoco.MjData( self.model )
        for i in range( num_threads ):
            self.data_list.append( copy.copy( tmp ) )
        del tmp

        self.num_envs = num_envs
        self.num_threads = num_threads

        self._config = config
        self._ctrl_dt = config.ctrl_dt
        self._sim_dt = config.sim_dt
        self._nstep = int(round(self._ctrl_dt / self._sim_dt))
        self._episode_length = config.episode_length

    @property
    def dt(self):
        """Control timestep for the environment."""
        return self._ctrl_dt

    @property
    def sim_dt(self):
        """Simulation timestep for the environment."""
        return self._sim_dt
    
    @property
    def action_size(self):
        """Size of the action space."""
        return 5
        
    @property
    def observation_size(self):
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        return obs.shape[-1]
    
    @property
    def goal_size(self):
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        goal = abstract_state.info["achieved_goal"]
        return goal.shape[-1]
    
    def _init(self, xml_path, config):

        # prepare spec and add objects to the spec
        spec = self._prepare_spec(xml_path)
        spec, self._object_names = self._add_objects(spec, config.num_cubes)

        # compile spec and create mujoco model and data
        self.model = spec.compile()
        self.model.opt.timestep = config.sim_dt
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:12] = [0, 0, 0.07, 0, 0.0029, 0.00044, 0.0053, -0.0077, 0.0029, 0.00044, 0.0053, -0.0077]
        self.data.ctrl = np.array([0, 0, 0, 0.07, 0])

        mujoco.mj_step(self.model, self.data, nstep=int(round(config.ctrl_dt / config.sim_dt)))
        
        # set critical damping for arm joints
        for jn in _ARM_JOINTS:
            ji = self.model.joint(jn).id

            joint_mass = self.model.dof_M0[ self.model.jnt_dofadr[ji] ]
            stiffness = self.model.actuator(jn).gainprm[0]
            damping = 2 * math.sqrt(joint_mass * stiffness)

            self.model.joint(jn).damping = damping

        # get dimensions
        self._sensor_dim = self.model.nsensordata
        self._pstate_dim = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self._qpos_dim = self.model.nq
        self._qvel_dim = self.model.nv
        self._ctrl_dim = self.model.nu

        # get mocap ids
        self._mocap_targets = np.array([
            self.model.body( f"target_mocap_{i}" ).mocapid[0]
            for i in range( config.num_cubes )
        ])
        self._mocap_targets_geom = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"target_mocap_{i}")
            for i in range( config.num_cubes )
        ])

        # get start indices in qpos and qvel
        self._objs_qposadr = np.array([
            self.model.jnt_qposadr[ self.model.body(obj_name).jntadr[0] ]
            for obj_name in self._object_names
        ])
        self._objs_qveladr = np.array([
            self.model.jnt_dofadr[self.model.body(obj_name).jntadr[0]]
            for obj_name in self._object_names
        ])
        self._fingers_qposadr = np.array([
            self.model.jnt_qposadr[ self.model.joint(joint_name).id ]
            for joint_name in ['left_driver_joint', 'right_driver_joint']
        ])

        # get object pos and quat indices in physics_state
        self._objs_pos_physadr = np.concatenate([
            1 + obj_adr + np.arange(3)
            for obj_adr in self._objs_qposadr
        ])
        self._objs_quat_physadr = np.concatenate([
            1 + obj_adr + 3 + np.arange(4)
            for obj_adr in self._objs_qposadr
        ])

        # get sensor ids, addresses, dimensions and sensor data dims
        self._gripper_pos_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "pinch_framepos_sensor")
        self._gripper_quat_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "pinch_framequat_sensor")
        self._gripper_linvvel_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "pinch_framelinvel_sensor")
        
        self._gripper_pos_sensor_adr, self._gripper_pos_sensor_dim = self.model.sensor_adr[self._gripper_pos_sensor_id], self.model.sensor_dim[self._gripper_pos_sensor_id]
        self._gripper_quat_sensor_adr, self._gripper_quat_sensor_dim = self.model.sensor_adr[self._gripper_quat_sensor_id], self.model.sensor_dim[self._gripper_quat_sensor_id]
        self._gripper_linvvel_sensor_adr, self._gripper_linvvel_sensor_dim = self.model.sensor_adr[self._gripper_linvvel_sensor_id], self.model.sensor_dim[self._gripper_linvvel_sensor_id]
           
        # get constants
        self._init_q = jnp.array(self.data.qpos, dtype=jnp.float32)
        self._init_v = jnp.array(self.data.qvel, dtype=jnp.float32) * 0
        self._init_ctrl = jnp.array(self.data.ctrl, dtype=jnp.float32)
        self._init_physics_state = jnp.concatenate( [jnp.zeros(1, dtype=jnp.float32), self._init_q, self._init_v] )
        self._init_sensor_data = jnp.array( self.data.sensordata,  dtype=jnp.float32)
        self._init_gripper_quat_inverse = jnp.array([0., 0., -1., 0.])          # gripper's frame is rotated by 180 degrees around the global y axis
        
        # set action scale.
        self._action_scale =  np.array([config.action_scale.finger_scale] + [config.action_scale.xyz_scale]*3 + [config.action_scale.yaw_scale])

        # set bounds       
        self._workspace_bounds = jnp.array([ [-0.05, -0.35, 0.00], [ 0.45,  0.35, 0.5 ] ])
        self._target_sampling_bounds = jnp.array([ [0.22, -0.1, 0.02], [0.32, +0.1, 0.02] ])
        self._ctrl_bounds = jnp.array( self.model.actuator_ctrlrange.T )

        # get task data
        task_data = np.load( epath.Path(__file__).resolve().parent / f'tasks/cube-{config.num_cubes}.npz')
        self._starts_data = jnp.array( task_data['starts'][config.task_id] )
        self._target_cube_masks_data = jnp.array( task_data['masks'][config.task_id] )
        self._target_goal_data = jnp.array( task_data['goals'][config.task_id][ self._target_cube_masks_data ] )
        self._num_task_cubes = np.sum(self._target_cube_masks_data)

        # task relevant mocap target ids
        self._task_mocap_targets = self._mocap_targets[ self._target_cube_masks_data ]

        # make non-target cubes invisible
        self.model.geom_rgba[ self._mocap_targets_geom[ ~task_data['masks'][config.task_id] ], -1] *= 0

    def _prepare_spec(self, xml_path):
        spec = mujoco.MjSpec.from_file(str(xml_path), assets=get_assets(xml_path))
        
        # pre-compile
        base_body = spec.body('base_mount')
        base_body.pos, base_body.quat = np.array([0., 0., 0.12]), np.array([0.0, 0.707106, 0.707106, 0.0])

        for name, dof in _FOREARM_DOFS.items():
            base_body.add_joint(
                name=name,
                type=dof.joint_type,
                axis=dof.axis,
                range=dof.range,
            )
            act_tx = spec.add_actuator(
                name=name,
                target=name,
                trntype=dof.transmission,
                ctrlrange=dof.range,
            )
            act_tx.set_to_position(kp=dof.stiffness)

        spec.stat.center = np.array([0.4, 0.0 , 0.4])
        spec.stat.extent = 1.2
        spec.visual.global_.elevation = -30.0
        spec.visual.global_.azimuth = 180

        return spec

    def _add_objects(self, spec, num_cubes):
        object_names = []
        for i in range(num_cubes):

            # stacked placement
            x = 0.1
            y = 0.0
            z = 0.02 * (2*i + 1)

            yaw = np.random.uniform(-1, 1) * np.pi
            quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

            body = spec.worldbody.add_body(
                name=f"block_{i}",
                pos=[x, y, z],
                quat=quat,
            )
            body.add_freejoint(name=f"block_joint_{i}",)
            body.add_geom(
                name=f"block_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                contype=3,   
                conaffinity=1,
                solref=[0.004, 1],
                size=[0.02, 0.02, 0.02],
                rgba=_CUSTOM_COLORS[i % len(_CUSTOM_COLORS)],
                density=1240,
            )

            # adding target position for cube
            body = spec.worldbody.add_body(
                name=f"target_mocap_{i}",
                mocap=True,
                pos=[x, y, z],
                quat=quat,   
            )
            body.add_geom(
                name=f"target_mocap_{i}",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0.01, 0.01],
                rgba=_CUSTOM_COLORS[i % len(_CUSTOM_COLORS)] - np.array([0.0, 0.0, 0.0, 0.8]),
                contype=0,
                conaffinity=0,
            )

            object_names.append(f"block_{i}")

        return spec, object_names
       
    def reset(self, rng: jax.Array):
        rng, rng_box, rng_target, rng_starts = jax.random.split(rng, 4)

        _starts_data = jax.random.permutation(rng_starts, self._starts_data, axis=0)

        object_pos = (
            jax.random.uniform(
                rng_box,
                (self._config.num_cubes, 3),
                minval=_starts_data[:, 0],
                maxval=_starts_data[:, 1],
            )
        )
        achieved_goal = object_pos[ self._target_cube_masks_data ].reshape(-1)
        object_pos = object_pos.reshape(-1)
        object_quat = jnp.tile(jnp.array([1,0,0,0], dtype=jnp.float32), (self._config.num_cubes, 1)).reshape(-1)

        target_pos = (
            jax.random.uniform(
                rng_target,
                (1, 3),
                minval=self._target_sampling_bounds[0],
                maxval=self._target_sampling_bounds[1],
            )
        )
        target_object_pos = ( target_pos + self._target_goal_data ).reshape(-1)
        target_object_quat = jnp.tile(jnp.array([1,0,0,0], dtype=jnp.float32), (self._num_task_cubes, 1)).reshape(-1)

        # initialize physics state and sensordata and set initial object position
        init_physics_state = (
            self._init_physics_state            
            .at[self._objs_pos_physadr] 
            .set(object_pos)
        )
        init_physics_state = (
            init_physics_state              
            .at[self._objs_quat_physadr] 
            .set(object_quat)
        )
        init_sensor_data = self._init_sensor_data
        init_ctrl = self._init_ctrl

        metrics = {
            "easy_success": jnp.array(0.0, dtype=float),
            "success": jnp.array(0.0, dtype=float),
            "out_of_bounds": jnp.array(0.0, dtype=float),
            "obj_lifted": jnp.array(0.0, dtype=float),
            "obj_moved": jnp.array(0.0, dtype=float),
            "obj_goal_dist": jnp.array(0.0, dtype=float),
        }
        info = {
            "rng": rng, 
            "achieved_goal": achieved_goal,
            "target_goal": target_object_pos,
            "target_mask": self._target_cube_masks_data,
            "target_mocap_pos": target_object_pos, 
            "target_mocap_quat": target_object_quat,
        }

        # calculate observation
        obs = self.get_obs(init_physics_state, init_sensor_data, init_ctrl, info)[0]
        
        # initial rewards and done
        if self._config.permutation_invariant_reward:
            reward, reward_info = self.get_permutation_invariant_reward_from_obs(init_physics_state, init_sensor_data, info)
        else:
            reward, reward_info = self.get_permutation_variant_reward_from_obs(init_physics_state, init_sensor_data, info)
        done, out_of_bounds = self.get_termination(init_physics_state, init_sensor_data)
        metrics.update(
            out_of_bounds=out_of_bounds.astype(float), 
            **reward_info,
        )
    
        state = State(init_physics_state, init_sensor_data, init_ctrl, obs, reward, done, metrics, info)
        return state
    
    def get_obs(self, physics_state, sensor_data, ctrl, info):

        gripper_pos = sensor_data[self._gripper_pos_sensor_adr: self._gripper_pos_sensor_adr+self._gripper_pos_sensor_dim]
        gripper_quat = sensor_data[self._gripper_quat_sensor_adr: self._gripper_quat_sensor_adr+self._gripper_quat_sensor_adr]
        gripper_quat = mjx_math.quat_mul(gripper_quat, self._init_gripper_quat_inverse)
        gripper_linvel = sensor_data[self._gripper_linvvel_sensor_adr:self._gripper_linvvel_sensor_adr+self._gripper_linvvel_sensor_dim]

        obj_pos = physics_state[1:][self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos[ self._target_cube_masks_data ].reshape(-1,)
        obj_pos = obj_pos.reshape(-1,)
        obj_quat = physics_state[1:][(self._objs_qposadr + 3)[:, None] + np.arange(4)].reshape(-1,)
        obj_linvel = physics_state[1 + self._qpos_dim:][self._objs_qveladr[:, None] + np.arange(3)].reshape(-1,)
        obj_angvel = physics_state[1 + self._qpos_dim:][(self._objs_qveladr + 3)[:, None] + np.arange(3)].reshape(-1,)

        finger_pos = physics_state[1:][self._fingers_qposadr]

        obs = jnp.concatenate([
            gripper_pos,
            gripper_quat,
            gripper_linvel,
            obj_pos,
            obj_quat,
            obj_linvel,
            obj_angvel,
            finger_pos,
        ])

        info.update({
            "achieved_goal": achieved_goal, 
        })

        return obs, info
    
    def get_permutation_invariant_reward_from_obs(self, physics_state, sensor_data, info):
        obj_pos = physics_state[1:][self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos[ self._target_cube_masks_data ]
        obj_linvel = physics_state[1 + self._qpos_dim:][self._objs_qveladr[:, None] + np.arange(3)]

        target_goal = info["target_goal"].reshape((self._num_task_cubes, -1))

        obj_target_pos_squared_pairwise_err = jnp.sum( (achieved_goal[None, :, :] - target_goal[:, None, :]) ** 2, axis=-1)
        cube_ids, target_ids = optax.assignment.hungarian_algorithm( obj_target_pos_squared_pairwise_err )
        obj_target_pos_err = jnp.sqrt( obj_target_pos_squared_pairwise_err[cube_ids, target_ids] )

        obj_lifted = jnp.sum( obj_pos[:, 2] > 0.05 ).astype(float)
        obj_moved = jnp.any( obj_linvel > 0.001 ).astype(float)
        
        reward = jnp.sum(1 - jnp.tanh(self._config.reward_sensitivity * obj_target_pos_err))

        success = jnp.all(obj_target_pos_err < self._config.success_threshold).astype(float)
        easy_success = jnp.all(obj_target_pos_err < self._config.easy_success_threshold).astype(float)

        reward_info = {
            "success": success,
            "easy_success":  easy_success,
            "obj_lifted": obj_lifted,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum( obj_target_pos_err ),
        }

        return reward, reward_info
    
    def get_permutation_variant_reward_from_obs(self, physics_state, sensor_data, info):
        obj_pos = physics_state[1:][self._objs_qposadr[:, None] + np.arange(3)]
        achieved_goal = obj_pos[ self._target_cube_masks_data ]
        obj_linvel = physics_state[1 + self._qpos_dim:][self._objs_qveladr[:, None] + np.arange(3)]

        target_goal = info["target_goal"].reshape((self._num_task_cubes, -1))
            
        obj_target_pos_err = jnp.linalg.norm(target_goal - achieved_goal, axis=-1)

        obj_lifted = jnp.sum( obj_pos[:, 2] > 0.05 ).astype(float)
        obj_moved = jnp.any( obj_linvel > 0.001 ).astype(float)
        
        reward = jnp.sum(1 - jnp.tanh(self._config.reward_sensitivity * obj_target_pos_err))

        success = jnp.all(obj_target_pos_err < self._config.success_threshold).astype(float)
        easy_success = jnp.all(obj_target_pos_err < self._config.easy_success_threshold).astype(float)

        reward_info = {
            "success": success,
            "easy_success":  easy_success,
            "obj_lifted": obj_lifted,
            "obj_moved": obj_moved,
            "obj_goal_dist": jnp.sum( obj_target_pos_err ),
        }

        return reward, reward_info
    
    def get_termination(self, physics_state, sensor_data):
        obj_pos = physics_state[1:][self._objs_qposadr[:, None] + np.arange(3)]

        out_of_bounds = 1 - ( jnp.all((obj_pos >= self._workspace_bounds[0]) & jnp.all(obj_pos <= self._workspace_bounds[1])) )
        termination = out_of_bounds | jnp.isnan(physics_state).any() | jnp.isnan(sensor_data).any()
        termination = ( termination * self._config.early_termination ).astype(float)
        termination = termination.astype(float)
        return termination, out_of_bounds

    def pre_step(self, state, action):
        new_ctrl = action * self._action_scale + state.ctrl
        new_ctrl = jnp.clip(new_ctrl, self._ctrl_bounds[0], self._ctrl_bounds[1])
        state = state.replace( ctrl = new_ctrl )
        return state
    
    def step(self, state, action):
        def env_rollout(physics_state, ctrl):
                        
            # initialize dummy physics state and sensor data
            output_physics_state = np.empty((self.num_envs, self._nstep, self._pstate_dim))
            output_sensor_data = np.empty((self.num_envs, self._nstep, self._sensor_dim))
            
            # rollout
            rollout.rollout(
                self.model_list,
                self.data_list,
                physics_state,
                ctrl,
                nstep=self._nstep,
                state=output_physics_state,
                sensordata=output_sensor_data,
                skip_checks=True,
                persistent_pool=True,
            )
        
            return output_physics_state[:, -1], output_sensor_data[:, -1]

        output_physics_state, output_sensor_data = jax.experimental.io_callback( env_rollout, (state.physics_state, state.sensordata),  state.physics_state, jnp.tile( jnp.expand_dims(state.ctrl, 1 ), reps=(1, self._nstep, 1) ) )
        output_physics_state = jax.device_put(output_physics_state)
        output_sensor_data = jax.device_put(output_sensor_data)

        return output_physics_state, output_sensor_data
    
    def post_step(self, state, physics_state, sensor_data):
        obs, info = self.get_obs(physics_state, sensor_data, state.ctrl, state.info)

        if self._config.permutation_invariant_reward:
            reward, reward_info = self.get_permutation_invariant_reward_from_obs(physics_state, sensor_data, info)
        else:
            reward, reward_info = self.get_permutation_variant_reward_from_obs(physics_state, sensor_data, info)
        done, out_of_bounds = self.get_termination(physics_state, sensor_data)

        state.metrics.update(
            out_of_bounds=out_of_bounds.astype(float), 
            **reward_info,
        )
        
        #update state of the environment
        state = State(physics_state, sensor_data, state.ctrl, obs, reward, done, state.metrics, info)
        
        return state
    
    def render(
        self,
        state : State,
        height: int = 480,
        width: int = 640,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
    ):
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        camera = camera or -1
        def get_image(state, modify_scn_fn=None) -> np.ndarray:
            d = mujoco.MjData(self.model)

            d.qpos, d.qvel = state.physics_state[1:][:self.model.nq], state.physics_state[1:][self.model.nq: self.model.nq + self.model.nv]            
            d.mocap_pos[self._task_mocap_targets], d.mocap_quat[self._task_mocap_targets] = state.info['target_mocap_pos'].reshape(self._num_task_cubes, 3), state.info['target_mocap_quat'].reshape(self._num_task_cubes, 4)
            mujoco.mj_forward(self.model, d)
            renderer.update_scene(d, camera=camera, scene_option=scene_option)
            if modify_scn_fn is not None:
                modify_scn_fn(renderer.scene)
            return renderer.render()

        out = get_image(state)
        renderer.close()

        return out
    
    def render_from_info(
        self,
        qpos, qvel, mocap_pos, mocap_quat,
        height: int = 480,
        width: int = 640,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
    ):
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        camera = camera or -1
        def get_image(qpos, qvel, mocap_pos, mocap_quat) -> np.ndarray:
            d = mujoco.MjData(self.model)
            d.qpos, d.qvel = qpos, qvel
            d.mocap_pos[self._task_mocap_targets], d.mocap_quat[self._task_mocap_targets] = mocap_pos.reshape(self._num_task_cubes, 3), mocap_quat.reshape(self._num_task_cubes, 4)
            mujoco.mj_forward(self.model, d)
            renderer.update_scene(d, camera=camera, scene_option=scene_option)
            return renderer.render()

        out = get_image(qpos, qvel, mocap_pos, mocap_quat)
        renderer.close()

        return out