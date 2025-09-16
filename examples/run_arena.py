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
    return hanabi_v5.env(colors=2, ranks=5, players=2, hand_size=2, max_information_tokens=3, max_life_tokens=1, observation_type='card_knowledge')
def UndercoverEnv(render_mode, **kwargs):
    return undercover_v1.env(render_mode=None,undercover_mode="semi") #random, semi, fixed
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

           

def run_arena(
    render: bool = False,
    parallel: bool = True,
    seed=0,
    total_games: int = 10,
    max_game_onetime: int = 50,
    use_tqdm: bool = True,
    game_envs: str = "",
    engine: str = "",
    models: list[str] = []
):
    """
    Args:
        render (bool): 是否渲染游戏界面。
        parallel (bool): 是否使用并行运行。
        seed (int): 随机种子。
        total_games (int): 期望的总游戏局数。
        max_game_onetime (int): 单次运行的最大游戏局数。
        use_tqdm (bool): 是否使用进度条。
        game_envs : 游戏环境。
    """
    
    # 初始化或加载结果
    results_path = "result.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    idx = 0
    games_played = 0
    env_name = game_envs

        
    agents = {}
    for i, model in enumerate(models):
        agent_name = f"{model}_{i}"
        agents[agent_name]=LocalAgent(env_name,model,engine)


    while games_played < total_games:
        idx += 1
        
        # 确保每个 llm 实例的键都是唯一的 gpt-4o-mini deepseek-r1

        
        env_wrappers = [RecordWinner]
        
        # Arena 初始化，只创建一次
        arena = make_arena(
            env_name, 
            env_wrappers=env_wrappers, 
            use_tqdm=use_tqdm, 
            dispatch_func=default_dispatch_func, 
            dispatch_func_multi=default_dispatch_func_multi
        )
        
        # 创建带有唯一键的agents字典
        # agents = {
        #     f"{llm}_{i}": LocalAgent(f"../selfplay/opponent_templates/{env_name}/{llm}") 
        #     for i, llm in enumerate(llms)
        # }

        # 从保存的结果中加载 Trueskill 评分
        if env_name in results:
            for llm_key in agents:
                if llm_key in results[env_name]:
                    agent_rating = results[env_name][llm_key]["rating"]
                    agents[llm_key].rating = trueskill.Rating(
                        mu=agent_rating["mu"], 
                        sigma=agent_rating["sigma"]
                    )

        # 确定本轮运行的游戏局数
        current_run_games = min(max_game_onetime, total_games - games_played)

        if current_run_games <= 0:
            break  # 避免多余的运行

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Running {current_run_games} games for env: {env_name}...")
        
        # 运行竞技场
        arena.reset(
            agents=agents,
            total_games=current_run_games,  # 使用本轮游戏局数
            max_game_onetime=current_run_games,
            seed=seed + idx,
        )
        
        round_result = arena.run(parallel=parallel)
        try:
            arena.close()
        except KeyboardInterrupt:
            arena.close() # 调用清理方法
            raise # 重新抛出异常，以便程序正常退出
        # 累积游戏局数
        games_played += current_run_games

        # 更新结果字典
        # 1. 确保环境键存在
        if env_name not in results:
            results[env_name] = {}
        
        # 2. 遍历本轮结果，更新或添加数据
        for llm_key, data in round_result.items():
            if llm_key in results[env_name]:
                # 更新现有数据
                before_games = results[env_name][llm_key]['total_games']
                after_games = data['total_games']
                total = max(before_games + after_games, 1)

                results[env_name][llm_key]['win_rate'] = (results[env_name][llm_key]['win_rate'] * before_games + data['win_rate'] * after_games) / total
                results[env_name][llm_key]['loss_rate'] = (results[env_name][llm_key]['loss_rate'] * before_games + data['loss_rate'] * after_games) / total
                results[env_name][llm_key]['draw_rate'] = (results[env_name][llm_key]['draw_rate'] * before_games + data['draw_rate'] * after_games) / total
                results[env_name][llm_key]['total_reward'] += data['total_reward']
                results[env_name][llm_key]['avg_reward'] = results[env_name][llm_key]['total_reward'] / total
                results[env_name][llm_key]['total_games'] += data['total_games']
                results[env_name][llm_key]['rating'] = data['rating']
            else:
                # 添加新数据
                results[env_name][llm_key] = data

        # print("="*50)
        # print(f"Current results for env '{env_name}':")
        # print(results[env_name])
        # print("="*50)
        
        # 每次循环结束后保存结果
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    print(f"Total games of {total_games} finished.")
    return results

if __name__ == "__main__":
    register_new_envs()
    game_env = "undercover_v1"  # "undercover_v1"
    
    Number_player=2 if game_env != "undercover_v1" else 5
    models = ["cot"] * Number_player



    # 调用 run_arena，让其在内部处理所有逻辑，并返回最终结果
    final_results = {}

    final_results = run_arena(
        render=False, 
        parallel=True, 
        seed=75, 
        total_games=2, # 这里的 total_games 会被正确使用
        max_game_onetime=1, # 建议将此值设置为较小，以便测试循环逻辑
        game_envs=game_env, # 只运行当前循环中的一个环境
        engine="xiaoai:gpt-5-nano@DEBUG", #@DEBUG
        models=models,
    )
    print(f"Final results for all runs: {final_results}")