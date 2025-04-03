import numpy as np
import mujoco

import argparse
import os
import pickle

import torch
from rsl_rl.runners import OnPolicyRunner

vel_cmd_global = np.array([0,0,0], dtype=np.float32)

class SimpleMujocoEnv:
    def __init__(self, model_path):
        # Load the MuJoCo model using the new official API
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # Define dimensions: observation is 27-dim, action is 6-dim.
        self.num_privileged_obs = 27
        self.num_obs = 27
        self.num_actions = 6
        self.num_envs = 1

        # Scaling factors for sensor data.
        self.obs_scales = {
            "ang_vel": np.array([0.25, 0.25, 0.25], dtype=np.float32),
            "dof_pos": np.ones(6, dtype=np.float32),
            "dof_vel": np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32),
        }
        self.commands_scale = 1.0

        # Default joint positions for the 6 joints.
        self.default_dof_pos = np.zeros(6, dtype=np.float32)

        # Placeholders for sensor readings.
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.commands = torch.zeros(3, dtype=torch.float32)
        self.dof_pos = np.zeros(6, dtype=np.float32)
        self.dof_vel = np.zeros(6, dtype=np.float32)
        self.actions = np.zeros(6, dtype=np.float32)

        self.gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro_imu_torso")
        self.acc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer_imu_torso")
        #self.acc_id = mujoco._mj_model.site("imu_torso").id

        # build the sensordata index map
        self.sensor_map = {}
        index = 0
        for sensor_id in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
            dim = self.model.sensor_dim[sensor_id]
            self.sensor_map[name] = {
                "id": sensor_id,
                "dim": dim,
                "start_index": index,
                "end_index": index + dim,
                "type": self.model.sensor_type[sensor_id],
            }
            index += dim

    def get_sensor_value(self, name):
        meta = self.sensor_map[name]
        start, end = meta["start_index"], meta["end_index"]
        return self.data.sensordata[start:end].copy()

    def step(self, action):
        # Apply the 6-dimensional action
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # --- Sensor Readouts ---
        # Base angular velocity (3 dims): adjust the slice if needed.
        gyro = self.data.sensordata[self.gyro_id:self.gyro_id+3].copy()
        self.base_ang_vel = torch.tensor(gyro, dtype=torch.float32)
        # Projected gravity (3 dims): here we assume a fixed gravity vector.
        grav =   self.data.sensordata[self.acc_id:self.acc_id+3].copy()
        self.projected_gravity = torch.tensor(grav, dtype=torch.float32)
        # Joint (DOF) positions: assuming indices 7 to 12 correspond to your 6 joints.
        self.dof_pos = torch.tensor(self.data.qpos[7:13], dtype=torch.float32)
        # Joint velocities: assuming indices 6 to 12 correspond to joint velocities.
        self.dof_vel = torch.tensor(self.data.qvel[6:12], dtype=torch.float32)
        # Record the applied action.
        self.actions = action

        # --- Build the Observation ---
        # Observation vector components:
        # - 3 dims: base angular velocity (scaled)
        # - 3 dims: projected gravity
        # - 3 dims: command inputs (scaled)
        # - 6 dims: joint positions (offset and scaled)
        # - 6 dims: joint velocities (scaled)
        # - 6 dims: last actions
        default_pos=np.array([0.6,-0.18,-0.9,-0.6,0.18,0.9], dtype=np.float32)

        obs = np.concatenate([
            self.base_ang_vel * self.obs_scales["ang_vel"],                    # 3 dims
            self.projected_gravity,                                            # 3 dims
            self.commands * self.commands_scale,                               # 3 dims
            (self.dof_pos - default_pos) * self.obs_scales["dof_pos"],  # 6 dims
            self.dof_vel * self.obs_scales["dof_vel"],                          # 6 dims
            self.actions,                                                      # 6 dims
        ])
        # Total dims: 3+3+3+6+6+6 = 27
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        reward = 0.0
        done = False
        info = {}
        return obs_tensor, reward, done, info

    def reset(self):
        # Reset the simulation data to the initial state.
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id == -1:
            raise ValueError("Keyframe 'home' not found in the model!")
        # Reset the simulation state to the keyframe "home"
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        # Optionally, perform a few simulation steps to allow the state to settle.
        for _ in range(2):
            mujoco.mj_step(self.model, self.data)
        # Return the initial observation.
        obs, _, _, _ = self.step(torch.zeros(6, dtype=torch.float32))
        dummy = 0
        return obs, dummy

    def render(self):
        # Use the official MuJoCo viewer.
        if self.viewer is None:
            from mujoco import viewer
            # launch_passive creates a viewer that synchronizes with the simulation.
            self.viewer = viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()

    def close(self):
        # Close the viewer if it exists.
        if self.viewer is not None:
            self.viewer = None

# Example usage:
if __name__ == "__main__":

    env = SimpleMujocoEnv("/home/ps/Documents/Genesis/genesis/assets/xml/bai_xml/scene.xml")  # Ensure your XML model file is in the current directory.

    parser = argparse.ArgumentParser()
    # modified name
    parser.add_argument("-e", "--exp_name", type=str, default="bpd-walking-0327-alldir")
    parser.add_argument("--ckpt", type=int, default=8000)
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    # cpu device at run time
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cpu")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cpu")

    obs,_ = env.reset()
    print("Initial observation:", obs)

    with torch.no_grad():

        for i in range(100):

            actions = policy(obs)
            actions = torch.zeros(6, dtype=torch.float32)

            #print(actions[:])
            index_order = torch.tensor([0, 2, 4, 1, 3, 5])
            action_reorder = actions[index_order]
            target_dof_pos = action_reorder*0.5  + torch.tensor([0.6,-0.18,-0.9,-0.6,0.18,0.9], dtype=torch.float32)
            obs, rews, dones, infos = env.step(target_dof_pos)

            #print(f"Step {i+1} observation:", obs)
            env.render()

            print("Gravity:", env.model.opt.gravity)
            grav = env.get_sensor_value("acc_static").copy()
            print(grav)

            site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "imu_torso")
            R = env.data.site_xmat[site_id].reshape(3, 3)  # World-to-local rotation matrix
            gravity_world = np.array([0, 0, -9.81])
            gravity_in_imu_frame = R.T @ gravity_world  # project world gravity into local frame
            print("Expected accelerometer reading (gravity in local frame):", gravity_in_imu_frame)

            grav_body2world = R @ grav
            print(grav_body2world)


        env.close()
