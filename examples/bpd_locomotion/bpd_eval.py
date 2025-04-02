import argparse
import os
import pickle

import torch
from bpd_env import BpdEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

# lcm external setup
import lcm
from lcm_t.exlcm import twist_t
import select

import numpy as np

vel_cmd_global = np.array([0,0,0], dtype=np.float32)

def lcm_handle(channel, data):
    msg = twist_t.decode(data)
    print("Received message on channel \"%s\"" % channel)
    print("   x   = %s" % str(msg.x_vel[0]))
    print("   y    = %s" % str(msg.y_vel[0]))
    print("   orientation = %s" % str(msg.omega_vel[0]))
    print("")

    vel_cmd_global[0] = msg.x_vel[0]
    vel_cmd_global[1] = msg.y_vel[0]
    vel_cmd_global[2] = msg.omega_vel[0]


def main():
    parser = argparse.ArgumentParser()
    # modified name
    parser.add_argument("-e", "--exp_name", type=str, default="bpd-walking-0327-alldir")
    parser.add_argument("--ckpt", type=int, default=6000)
    args = parser.parse_args()

    gs.init() # genesis

    lc = lcm.LCM()
    subscription = lc.subscribe("TWIST_T", lcm_handle)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = BpdEnv(
        num_envs=32, # how many envs to eval
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    # modify simtime for eval rollout
    env.max_episode_length = 1e4/env.dt

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    # Set lc.fileno() to non-blocking mode (assuming it's a socket or similar)
    fd = lc.fileno()
    os.set_blocking(fd, False)  # Make the file descriptor non-blocking

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            rfds, wfds, efds = select.select([fd], [], [], 0.001)  # Non-blocking with timeout=0
            if rfds:
                lc.handle()

            envs_idx = 0
            env.commands[envs_idx, :] = torch.tensor(vel_cmd_global, dtype=torch.float32, device=env.commands.device)
            print(env.commands[envs_idx, :])

            actions = policy(obs)
            print(actions[envs_idx, :])
            obs, _, rews, dones, infos = env.step(actions)




if __name__ == "__main__":
    main()
