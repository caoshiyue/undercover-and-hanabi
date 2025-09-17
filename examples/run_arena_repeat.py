##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2025-09-17 13:06:33
## 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The OpenRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""
from openrl.arena import make_arena
from openrl.envs.wrappers.pettingzoo_wrappers import RecordWinner
from openrl.selfplay.selfplay_api.opponent_model import BattleResult
from openrl.arena.agents.base_agent import BaseAgent

from pettingzoo.classic import connect_four_v3
from pettingzoo.classic import chess_v6
from pettingzoo.classic import go_v5
from pettingzoo.classic import texas_holdem_no_limit_v6
from pettingzoo.classic import hanabi_v5
from openrl.envs.bargain_env import bargain_v1
from openrl.envs.bid_env import bid_v1
from openrl.envs.undercover_env import undercover_v1
from openrl.envs.PettingZoo.registration import register
import importlib
from pathlib import Path
import json
import random
import numpy as np
import os
import requests
import trueskill
import time
from joblib import Parallel, delayed
import statistics

# baseagent 适配器
class LocalAgent(BaseAgent):
    def __init__(self, env_name,model_name,engine):
        super().__init__()
        self.env_name=env_name
        self.model_name=model_name
        self.engine=engine

    def _new_agent(self):
        opponent_module = importlib.import_module(f"examples.selfplay.opponent_templates.{ self.env_name}.opponent", package=__package__)
        OpponentClass = getattr(opponent_module, "Opponent")
        # 构造无用但保留的三个参数
        opponent_id = self.model_name 
        opponent_path = f"./selfplay/opponent_templates/{self.env_name}"  # 占位 examples/selfplay/opponent_templates/undercover_v1
        with open(f"./selfplay/opponent_templates/{self.env_name}/info.json", 'r') as f:
            opponent_info = json.load(f)

        # 初始化 Opponent
        return OpponentClass(
            opponent_id=opponent_id,
            opponent_path=opponent_path,
            opponent_info=opponent_info,
            agent_type=self.model_name,   # 关键：把 model 作为 agent_type
            agent_params={"model": self.engine},  # 可以传更多参数
        )

        

def ConnectFourEnv(render_mode, **kwargs):
    return connect_four_v3.env(render_mode=None)
def ChessEnv(render_mode, **kwargs):
    return chess_v6.env(render_mode=None)
def GoEnv(render_mode, **kwargs):
    return go_v5.env(board_size = 5)
def TexasEnv(render_mode, **kwargs):
    return texas_holdem_no_limit_v6.env(render_mode=None)
def BargainEnv(render_mode, **kwargs):
    return bargain_v1.env(render_mode=None)
def BidEnv(render_mode, **kwargs):
    return bid_v1.env(render_mode=None)
def HanabiEnv(render_mode, **kwargs):
    return hanabi_v5.env(colors=2, ranks=5, players=2, hand_size=2, max_information_tokens=6, max_life_tokens=3, observation_type='card_knowledge') #! 注意，原始colors=2, ranks=5, players=2, hand_size=2, max_information_tokens=3, max_life_tokens=1,
def UndercoverEnv(render_mode, **kwargs):
    return undercover_v1.env(render_mode=None,undercover_mode="fixed") #random, semi, fixed
def register_new_envs():
    register("connect_four_v3", ConnectFourEnv)
    register("chess_v6", ChessEnv)
    register("go_v5", GoEnv)
    register("texas_no_limit_v6", TexasEnv)
    register("hanabi_v5", HanabiEnv)
    register("bargain_v1", BargainEnv)
    register("bid_v1", BidEnv)
    register("undercover_v1", UndercoverEnv)

    return ["connect_four_v3","chess_v6","go_v5","texas_no_limit_v6","hanabi_v5"]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def default_dispatch_func(
    np_random: np.random.Generator,
    players,
    agent_names
) :
    assert len(players) == 2, "The number of players must be equal to 2."
    np_random.shuffle(agent_names)
    return dict(zip(players, agent_names[:2]))

def default_dispatch_func_multi(
    np_random: np.random.Generator,
    players,
    agent_names,
    num_agents = 5
) :
    assert len(players) == num_agents, f"The number of players must be equal to {num_agents}."
    np_random.shuffle(agent_names)
    return dict(zip(players, agent_names[:num_agents]))

           


def run_single_experiment(
    experiment_seed: int, 
    games_per_experiment: int, 
    game_env: str, 
    models: list[str], 
    engine: str,
    use_tqdm: bool = False
):
    """
    Runs a single, self-contained experiment of M games.

    Args:
        experiment_seed (int): The random seed for this specific experiment to ensure reproducibility.
        games_per_experiment (int): The number of games (M) to run in this experiment.
        game_env (str): The name of the game environment.
        models (list[str]): A list of model identifiers for creating agents.
        engine (str): The engine to be used by the agents.
        use_tqdm (bool): Whether to show a progress bar for this run.

    Returns:
        dict: The results dictionary from this single experiment.
    """
    print(f"[Seed: {experiment_seed}] Starting experiment with {games_per_experiment} games...")
    # 对新线程，再次注册环境
    register_new_envs()
    # 1. Initialize fresh agents for this experiment
    # This ensures no state is carried over from other experiments.
    agents = {}
    for i, model in enumerate(models):
        agent_name = f"{model}_{i}"
        agents[agent_name] = LocalAgent(game_env, model, engine)

    # 2. Initialize a fresh arena for this experiment
    env_wrappers = [RecordWinner]
    arena = make_arena(
        game_env,
        env_wrappers=env_wrappers,
        # Disable tqdm for parallel runs to avoid messy output, enable for the main progress bar
        use_tqdm=use_tqdm,
        dispatch_func=default_dispatch_func,
        dispatch_func_multi=default_dispatch_func_multi
    )

    # 3. Run the M games for this experiment
    arena.reset(
        agents=agents,
        total_games=games_per_experiment,
        max_game_onetime=games_per_experiment, # Run all games in one go
        seed=experiment_seed,
    )
    
    experiment_result = arena.run(parallel=False) # Parallelism is handled at the experiment level
    arena.close()
    
    return experiment_result

# --- Main Arena Runner ---

def run_arena_experiments(
    num_experiments: int = 10,       # N: How many times to repeat the experiment
    games_per_experiment: int = 100, # M: How many games in one experiment
    parallel: bool = True,
    seed: int = 0,
    game_env: str = "",
    engine: str = "",
    models: list[str] = []
):
    """
    Manages and runs N independent experiments, each with M games.

    Args:
        num_experiments (int): The total number of experiments to run (N).
        games_per_experiment (int): The number of games to play per experiment (M).
        parallel (bool): Whether to run the N experiments in parallel.
        seed (int): The base random seed. Each experiment will get a unique seed derived from this.
        game_env (str): The game environment.
        engine (str): The engine for the agents.
        models (list[str]): The models for the agents.
    """
    total_games = num_experiments * games_per_experiment
    print(f"Starting {num_experiments} experiments of {games_per_experiment} games each.")
    print(f"Total games to be played: {total_games}")
    
    start_time = time.time()
    all_results = []

    if parallel:
        # Use joblib to run experiments in parallel. n_jobs=-1 uses all available CPU cores.
        # Each call to run_single_experiment is one independent job.
        all_results = Parallel(n_jobs=-1)(
            delayed(run_single_experiment)(
                experiment_seed=seed + i,
                games_per_experiment=games_per_experiment,
                game_env=game_env,
                models=models,
                engine=engine,
                use_tqdm=False # TQDM is better handled at the top level
            ) for i in range(num_experiments)
        )
    else:
        # Run experiments serially (sequentially)
        for i in range(num_experiments):
            result = run_single_experiment(
                experiment_seed=seed + i,
                games_per_experiment=games_per_experiment,
                game_env=game_env,
                models=models,
                engine=engine,
                use_tqdm=True # Can use tqdm here since it's serial
            )
            all_results.append(result)
    
    end_time = time.time()
    print(f"Finished all {num_experiments} experiments in {end_time - start_time:.2f} seconds.")

    # --- 4. Aggregate the final statistics from all experiments ---
    final_stats = {}
    
    # Get the list of all agent keys from the first experiment's result
    if not all_results:
        print("No results to aggregate.")
        return {}
        
    agent_keys = all_results[0].keys()

    for agent_key in agent_keys:
        final_stats[agent_key] = {
            'win_rates': [res[agent_key]['win_rate'] for res in all_results],
            'avg_rewards': [res[agent_key]['avg_reward'] for res in all_results],
            # Add other metrics you want to track across experiments
        }

    # Calculate mean and standard deviation
    aggregated_results = {}
    for agent_key, stats in final_stats.items():
        aggregated_results[agent_key] = {
            'agent_name': agent_key,
            'num_experiments': num_experiments,
            'games_per_experiment': games_per_experiment,
            'win_rate_mean': statistics.mean(stats['win_rates']),
            'win_rate_std': statistics.stdev(stats['win_rates']) if num_experiments > 1 else 0,
            'avg_reward_mean': statistics.mean(stats['avg_rewards']),
            'avg_reward_std': statistics.stdev(stats['avg_rewards']) if num_experiments > 1 else 0,
        }
        
    # Save the final aggregated results
    results_path = f"results_{game_env}.json"
    with open(results_path, 'w') as f:
        json.dump(aggregated_results, f, indent=4)
        
    print(f"Final aggregated results saved to {results_path}")
    return aggregated_results


if __name__ == "__main__":
    register_new_envs()
    game_env = "bargain_v1" # undercover_v1  hanabi_v5
    
    Number_player = 5 if "undercover_v1" in game_env else 2
    models =["llm","cot"] # ["llm"] * Number_player

    # Call the main experiment runner
    final_results = run_arena_experiments(
        num_experiments=30,          # N: Run 5 separate experiments
        games_per_experiment=5,    # M: Each experiment consists of 10 games
        parallel=True,              # True  False
        seed=75,
        game_env=game_env,
        engine="xiaoai:gpt-5-nano", # @DEBUG  gpt-5-nano gpt-4o-mini
        models=models,
    )
    
    print("\n" + "="*50)
    print("Final Aggregated Results:")
    print(json.dumps(final_results, indent=4))
    print("="*50)
