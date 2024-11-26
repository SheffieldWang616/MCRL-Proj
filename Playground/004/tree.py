import gym

from minerl.env import _fake, _singleagent
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.mc import ALL_ITEMS
from typing import List

TIMEOUT = 8000
TREE_ITEMS = [
    [["acacia_log", "birch_log", "dark_oak_log", "jungle_log", "oak_log", "spruce_log"], 1, 16],
]


class TREEWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.rewarded_items = TREE_ITEMS
        self.timeout = self.env.task.max_episode_steps
        self.seen = {item: 0 for item in TREE_ITEMS[0][0]}
        self.num_steps = 0
        self.episode_over = False

    def step(self, action: dict):
        if self.episode_over:
            raise RuntimeError("Expected `reset` after episode terminated, not `step`.")
        observation, reward, done, info = super().step(action)

        for i, [item_list, rew, tar_num] in enumerate(self.rewarded_items):
            for item in item_list:
                item_num = observation["inventory"][item]
                if item_num > self.seen[item]:
                    reward += (item_num - self.seen[item]) * rew
                    self.seen[item] = item_num
                elif item_num < self.seen[item]:
                    reward -= (self.seen[item] - item_num) * rew * 2
                    self.seen[item] = item_num
                    
                if tar_num <= item_num:
                    done = True
        self.num_steps += 1
        if self.num_steps >= self.timeout:
            done = True
        self.episode_over = done
        return observation, reward, done, info

    def reset(self):
        self.seen = {item: 0 for item in TREE_ITEMS[0][0]}
        self.episode_over = False
        obs = super().reset()
        return obs


def _tree_gym_entrypoint(env_spec, fake=False):
    """Used as entrypoint for `gym.make`."""
    if fake:
        env = _fake._FakeSingleAgentEnv(env_spec=env_spec)
    else:
        env = _singleagent._SingleAgentEnv(env_spec=env_spec)

    env = TREEWrapper(env)
    return env

TREE_ENTRY_POINT = "minerl.herobraine.env_specs.tree:_tree_gym_entrypoint"

class TreeEnvSpec(HumanSurvival):
    r"""
    In this environment, the agent is required to obtain a gold block.
    The agent begins at a high elevation with a water bucket and a diamond pickaxe.
    
    Reward:
        - The agent receives a reward of 50 points for obtaining a gold block.
    """
    def __init__(self):
        super().__init__(
            name="Tree-v0",
            max_episode_steps=TIMEOUT * 10,
            # Hardcoded variables to match the pretrained models
            fov_range=[70, 70],
            resolution=[640, 360],
            gamma_range=[2, 2],
            guiscale_range=[1, 1],
            cursor_size_range=[16.0, 16.0]
        )

    def _entry_point(self, fake: bool) -> str:
        return TREE_ENTRY_POINT

    def create_observables(self) -> List[Handler]:
        return [
            handlers.POVObservation(self.resolution),
            handlers.FlatInventoryObservation(ALL_ITEMS)
        ]

    def create_monitors(self) -> List[TranslationHandler]:
        return []

    def create_server_world_generators(self) -> List[Handler]:
        return super().create_server_world_generators() + \
            [
            handlers.FlatWorldGenerator(generatorString="3;7,2x3,2;1;village", force_reset=True),
            # handlers.FlatWorldGenerator(force_reset=True),
            # generate a 3x3 square of obsidian high in the air and a gold block
            # somewhere below it on the ground
            # handlers.DrawingDecorator("""
            #     <DrawCuboid x1="0" y1="5" z1="-6" x2="0" y2="5" z2="-6" type="gold_block"/>
            #     <DrawCuboid x1="-2" y1="88" z1="-2" x2="2" y2="88" z2="2" type="obsidian"/>
            # """)
            ]
    
    def create_agent_start(self) -> List[Handler]:
        return super().create_agent_start() + \
            [
            handlers.AgentStartPlacement(90.0, 64.0, 264.0),
            handlers.SimpleInventoryAgentStart([
            {"type": "diamond_axe", "quantity": 1},
            ]),
            # handlers.SpawnInVillage()
            # handlers.AgentStartBreakSpeedMultiplier(5.0)
            ]
