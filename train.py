"""
Trains a policy using the assembly gym env and Stablebaseline for the RL algorithm
"""

import argparse
import datetime
import logging
from pathlib import Path
import yaml

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

# ---- project imports -------------------------------------------------------
from assembly_gym import BlockAssemblyGym
from utils.logger_utils import get_logger

ALGOS = {
    "maskppo": MaskablePPO,
    "ppo": PPO,
    "dqn": DQN
}

def load_hyperparams(path):
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def main():
    args = create_parser().parse_args()

    hyper = load_hyperparams(args.config)

    for k, v in hyper.items():
        if hasattr(args, k):
            setattr(args, k, v)        
        else:
            hyper[k] = v

    logger = get_logger(__name__)
    level = logging.INFO

    if args.debug:
        level = logging.DEBUG
        logger.setLevel(level)
        logger.debug("Debug logging is enabled.")
    

    logger.info(f"Device used: {args.device}")

    # Make save directory
    run_name = datetime.datetime.now().strftime("%m%d%H%M%S")
    run_dir = Path(args.logdir) / f"{args.task}_{args.algo}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Run Directory {run_dir}")

    def make_env(rank: int):
        "Factory needed by VecEnv constructors"
        def _init():
            env = BlockAssemblyGym(
                task=args.task,
                render=args.render and args.n_envs == 1,  # only render when 1 env
                level=level,
            )
            return Monitor(env)
        return _init

    '''env_fns = [make_env(i) for i in range(args.n_envs)]

    if args.n_envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns, start_method="spawn")'''

    if args.algo == "dqn":
        if args.n_envs != 1:
            raise ValueError("DQN only supports a single environment. Use --n-envs 1.")
        env = make_env(0)()
    else:
        env_fns = [make_env(i) for i in range(args.n_envs)]
        if args.n_envs == 1:
            env = DummyVecEnv(env_fns)
        else:
            env = SubprocVecEnv(env_fns, start_method="spawn")


    Algo = ALGOS[args.algo]
    if args.resume_model:
        logger.info(f"Resuming from {args.resume_model}")
        model = Algo.load(
            args.resume_model,
            env=env,                       
            device=args.device
        )
        # keep original tensorboard path if you want continuity
        #model.set_tensorboard_log(str(run_dir / "tb"))
    else:
        model = Algo(
            env=env,
            buffer_size=30000,
            tensorboard_log=str(run_dir / "tb"),
            device=args.device,
            policy="MlpPolicy", #adjust since for non-dqn this argument doesn't exist
            verbose=0,
            **{k: v for k, v in hyper.items() if k not in vars(args)}
        )

    chk_callback = CheckpointCallback(save_freq=args.save_freq, save_path=run_dir / "checkpoints", name_prefix="rl_model")
    #eval_env = DummyVecEnv([make_env(0)])
    eval_env = make_env(0)() if args.algo == "dqn" else DummyVecEnv([make_env(0)])
    if args.algo == "maskppo":
        eval_callback = MaskableEvalCallback(eval_env, best_model_save_path=run_dir / "best_model", eval_freq=args.save_freq,
                                    log_path=run_dir / "eval_logs", deterministic=True, render=False)
    else :
        eval_callback = EvalCallback(eval_env, best_model_save_path=run_dir / "best_model", eval_freq=args.save_freq,
                                    log_path=run_dir / "eval_logs", deterministic=True, render=False)
    
    model.learn(total_timesteps=args.timesteps, callback=[chk_callback, eval_callback],progress_bar=args.progress_bar)

    model.save(run_dir / "final_model")
    print(f"Training done -- model saved to {run_dir}")


def create_parser():
    parser = argparse.ArgumentParser(description="Train Rl policy on block-assembly task (SB3)")
    parser.add_argument("--task", choices=["bridge", "tower", "double_bridge"], default="bridge")
    parser.add_argument("--num-stories", type=int, default=2, help="difficulty setting for the chosen task")
    parser.add_argument("--algo", choices=list(ALGOS.keys()), default="maskppo")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--save-freq", type=int, default=10_000, help="checkpoint frequency (steps)")
    parser.add_argument("--logdir", default="runs", help="output directory")
    parser.add_argument("--device", default="cpu", help="auto, cpu, cuda")
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--progress-bar", action="store_true", help="Enable progress bar")
    parser.add_argument("--config", help="Path to YAML with hyper‑params")
    parser.add_argument("-m", "--resume-model", help="Path to *.zip model to continue training from")
    parser.add_argument("-n", "--n-envs", type=int, default=1, help="Number of environments to run in parallel")
    return parser

if __name__ == '__main__':
    main()