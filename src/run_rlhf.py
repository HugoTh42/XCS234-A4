import pathlib
try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
import stable_baselines3 as sb3
import torch
from tqdm import trange
from data import load_data
from util import export_plot
import importlib.util

use_submission = importlib.util.find_spec('submission') is not None
if use_submission:
  from submission import RewardModel

class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env, reward_fn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self._obs = obs.copy()
        return obs, info

    def step(self, action):
        next_obs, og_reward, terminated, truncated, info = self.env.step(action)
        reward = self.reward_fn.compute_reward(self._obs, action)
        info["og_reward"] = og_reward
        self._obs = next_obs.copy()
        return next_obs, reward, terminated, truncated, info


def evaluate(env, policy):
    model_return, og_return = 0, 0
    T = env.spec.max_episode_steps
    obs, _ = env.reset()
    for _ in range(T):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        model_return += reward
        og_return += info["og_reward"]
        if done:
            break
    return model_return, og_return


class EvalCallback(sb3.common.callbacks.BaseCallback):
    def __init__(self, eval_period, num_episodes, env, policy):
        super().__init__()
        self.eval_period = eval_period
        self.num_episodes = num_episodes
        self.env = env
        self.policy = policy

        self.original_returns = []
        self.learned_returns = []

    def _on_step(self):
        if self.n_calls % self.eval_period == 0:
            print(f"Evaluating after {self.n_calls} steps")
            model_returns, og_returns = [], []
            for _ in range(self.num_episodes):
                model_return, og_return = evaluate(self.env, self.policy)
                model_returns.append(model_return)
                og_returns.append(og_return)
            print(
                f"Model return: {np.mean(model_returns):.2f} +/- {np.std(model_returns):.2f}"
            )
            print(
                f"Original return: {np.mean(og_returns):.2f} +/- {np.std(og_returns):.2f}"
            )
            self.original_returns.append(np.mean(og_returns))
            self.learned_returns.append(np.mean(model_returns))

        # If the callback returns False, training is aborted early.
        return True


def main(args):
    output_path = pathlib.Path(__file__).parent.joinpath(
        "results_rlhf",
        f"Hopper-v4-rlhf-seed={args.seed}",
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_output = output_path.joinpath("model.zip")
    original_scores_output = output_path.joinpath("original_scores.npy")
    learned_scores_output = output_path.joinpath("learned_scores.npy")
    original_plot_output = output_path.joinpath("original_scores.png")
    learned_plot_output = output_path.joinpath("learned_scores.png")

    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    data = load_data(args.dataset_path)
    reward_model = RewardModel(
        obs_dim,
        action_dim,
        args.reward_model_hidden_dim,
        args.reward_min,
        args.reward_max,
    )

    print("Training reward model")
    for _ in trange(args.reward_model_steps):
        reward_model.update(data.sample(args.reward_model_batch_size))

    custom_reward_env = CustomRewardEnv(env, reward_model)
    agent = sb3.PPO("MlpPolicy", custom_reward_env, verbose=1)
    eval_callback = EvalCallback(
        args.eval_period,
        args.num_eval_episodes,
        custom_reward_env,
        lambda obs: agent.predict(obs)[0],
    )
    agent.learn(args.rl_steps, callback=eval_callback, progress_bar=True)

    # Log the results
    if not output_path.exists():
        output_path.mkdir(parents=True)
    agent.save(model_output)
    np.save(original_scores_output, eval_callback.original_returns)
    np.save(learned_scores_output, eval_callback.learned_returns)

    export_plot(
        eval_callback.original_returns,
        "Original returns",
        "Hopper-v4",
        original_plot_output,
    )
    export_plot(
        eval_callback.learned_returns,
        "Learned returns",
        "Hopper-v4",
        learned_plot_output,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--env-name", default="Hopper-v4")
    parser.add_argument(
        "--dataset-path",
        default=pathlib.Path(__file__).parent.joinpath("data", "prefs-hopper.npz"),
    )
    parser.add_argument("--reward-min", type=float, default=0.0)
    parser.add_argument("--reward-max", type=float, default=1.0)
    parser.add_argument("--reward-model-hidden-dim", type=int, default=64)
    parser.add_argument("--reward-model-steps", type=int, default=100000)
    parser.add_argument("--reward-model-batch-size", type=int, default=64)
    parser.add_argument("--rl-steps", type=int, default=1000000)
    parser.add_argument("--eval-period", type=int, default=1000)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
