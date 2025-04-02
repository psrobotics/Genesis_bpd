import argparse
import os
import pickle
import torch
import genesis as gs
from bpd_env import BpdEnv
from rsl_rl.runners import OnPolicyRunner

def export_policy_to_onnx(exp_name, ckpt, output_onnx_file):
    # Initialize genesis and set up the environment configurations
    gs.init()
    log_dir = f"logs/{exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}  # override reward scales if necessary

    # Create an evaluation environment (using one environment for simplicity)
    env = BpdEnv(
        num_envs=1,  # Use a single environment instance for export
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,  # disable visualization for export
    )
    # Modify simtime for a typical evaluation rollout
    env.max_episode_length = 1e4 / env.dt

    # Initialize the runner and load the trained checkpoint
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")
    policy.eval()  # Set the policy to evaluation mode

    # Reset the environment to get a dummy observation that matches your observation space
    obs, _ = env.reset()

    # Create a dummy input from the first observation in the batch
    # Adjust this line as needed based on the shape/type of obs
    dummy_input = torch.tensor(obs[0:1], dtype=torch.float32).to("cuda:0")

    # Export the policy to an ONNX model
    torch.onnx.export(
        policy,                   # The policy model to export
        dummy_input,              # Dummy input for tracing
        output_onnx_file,         # Output ONNX file name
        export_params=True,       # Include model parameters
        opset_version=11,         # ONNX version
        do_constant_folding=True, # Optimization step
        input_names=["obs"],      # Input tensor name
        output_names=["action"],  # Output tensor name
        dynamic_axes={
            "obs": {0: "batch_size"},
            "action": {0: "batch_size"}
        }  # Allow dynamic batch sizes
    )
    print(f"Exported policy to {output_onnx_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="bpd-walking-0327-alldir")
    parser.add_argument("--ckpt", type=int, default=6000)
    parser.add_argument("--output", type=str, default="exported_policy.onnx")
    args = parser.parse_args()
    
    export_policy_to_onnx(args.exp_name, args.ckpt, args.output)
