import pathlib
try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
import torch
from data import load_data
from tqdm import trange
from util import export_plot, standard_error
import importlib.util

use_submission = importlib.util.find_spec('submission') is not None
if use_submission:
  from submission import SFT, DPO


def evaluate(env, policy):
    total_reward = 0
    T = env.spec.max_episode_steps
    obs, _ = env.reset()
    for _ in range(T):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if done:
            break
    return total_reward


def get_batch(dataset, batch_size):
    obs1, obs2, act1, act2, label = dataset.sample(batch_size)
    obs = obs1[:, 0]
    assert torch.allclose(obs, obs2[:, 0])

    # Initialize assuming 1st actions preferred,
    # then swap where label = 1 (indicating 2nd actions preferred)
    actions_w = act1.clone()
    actions_l = act2.clone()
    swap_indices = label.nonzero()[:, 0]
    actions_w[swap_indices] = act2[swap_indices]
    actions_l[swap_indices] = act1[swap_indices]
    return obs, actions_w, actions_l


def main(args):
    output_path = pathlib.Path(__file__).parent.joinpath(
        "results_dpo",
        f"Hopper-v4-dpo-seed={args.seed}",
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_pretrained_output = output_path.joinpath("model_sft.pt")
    model_output = output_path.joinpath("model.pt")
    scores_output = output_path.joinpath("scores.npy")
    plot_output = output_path.joinpath("scores.png")

    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # DPO assumes preferences are strict, so we ignore the equally preferred pairs
    pref_data = load_data(args.dataset_path, strict_pref_only=True)
    segment_len = pref_data.sample(1)[0].size(1)

    print("Training SFT policy")
    sft = SFT(obs_dim, action_dim, args.hidden_dim, segment_len)
    for _ in trange(args.num_sft_steps):
        obs, actions_w, _ = get_batch(pref_data, args.batch_size)
        sft.update(obs, actions_w)

    print("Evaluating SFT policy")
    returns = [evaluate(env, sft.act) for _ in range(args.num_eval_episodes)]
    print(f"Return: {np.mean(returns):.2f} +/- {standard_error(returns):.2f}")

    print("Training DPO policy")
    dpo = DPO(
        obs_dim, action_dim, args.hidden_dim, segment_len, args.beta, lr=args.dpo_lr
    )
    dpo.net.load_state_dict(sft.net.state_dict())  # init with SFT parameters
    all_returns = []
    for step in trange(args.num_dpo_steps):
        obs, actions_w, actions_l = get_batch(pref_data, args.batch_size)
        dpo.update(obs, actions_w, actions_l, sft)
        if (step + 1) % args.eval_period == 0:
            print("Evaluating DPO policy")
            returns = [evaluate(env, dpo.act) for _ in range(args.num_eval_episodes)]
            print(f"Return: {np.mean(returns):.2f} +/- {standard_error(returns):.2f}")
            all_returns.append(np.mean(returns))

    # Log the results
    if not output_path.exists():
        output_path.mkdir(parents=True)
    with open(model_pretrained_output, "wb") as f:
        torch.save(sft, f)
    with open(model_output, "wb") as f:
        torch.save(dpo, f)
    np.save(scores_output, all_returns)
    export_plot(all_returns, "Returns", "Hopper-v4", plot_output)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--env-name", default="Hopper-v4")
    parser.add_argument(
        "--dataset-path",
        default=pathlib.Path(__file__).parent.joinpath("data", "prefs-hopper.npz"),
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-sft-steps", type=int, default=50000)
    parser.add_argument("--num-dpo-steps", type=int, default=50000)
    parser.add_argument("--dpo-lr", type=float, default=1e-6)
    parser.add_argument("--eval-period", type=int, default=1000)
    parser.add_argument("--num-eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())