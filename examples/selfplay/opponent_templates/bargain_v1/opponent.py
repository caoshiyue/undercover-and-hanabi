##
# Author:  
# Description:  
# LastEditors: Shiyuec
# LastEditTime: 2025-09-17 12:34:11
## 

import os 
from .llm_framework import *
from .env_opponent import env_Opponent

##
# Description 对接部分，按照环境要求修改
## 
class Opponent(env_Opponent):
    ##
    # Description 重载init
    ## 
    def __init__(self, agent_type="llm", agent_params=None, config_path=None, **kwargs):
        super().__init__(**kwargs)

        #  ---------- 这部分是固定的 ----------
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "llm_framework", "game.yaml")

        self.config = ConfigLoader(config_path)
        if agent_type not in AGENT_REGISTRY:
            raise ValueError(f"Unknown agent_type: {agent_type}. "
                            f"Available: {list(AGENT_REGISTRY.keys())}")

        AgentClass = AGENT_REGISTRY[agent_type]
        self.agent = AgentClass(self.config, **(agent_params or {}))
        #  ---------- 下面增加自定义逻辑 --------

    @async_adapter
    async def _request_action(self, message: str) -> str:
        ##
        # Description: 重载act,保持输入输出,执行原有act的逻辑
        ## 
        self.messages.append({"role":"user","content":message})
        answer = await self.agent.act(self.messages)
        answer=answer["answer"]
        # print(f'==========={self.player_name}===========')
        # print(answer)
        if self.need_history:
            self.messages.append({"role":"assistant","content":answer})
        else:
            self._reset_message()
        
        return answer
        


