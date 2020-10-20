# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""V-trace (IMPALA) learner for Google Research Football."""

from absl import app
from absl import flags

from seed_rl.agents.vtrace import learner
from seed_rl.common import actor
from seed_rl.common import common_flags  
from seed_rl.football import env
from seed_rl.football import networks
import tensorflow as tf
from collections import deque
import gym
import numpy as np
import sys
import importlib

FLAGS = flags.FLAGS

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')


# https://sites.google.com/view/rl-football/singleagent-team
class DifficultyWrapper(gym.Wrapper):
  def __init__(self, env):
    # Call the parent constructor, so we can access self.env later
    super(DifficultyWrapper, self).__init__(env)
    print(f"Initialized DifficultyWrapper {self.unwrapped._env._config._scenario_cfg.right_team_difficulty}", file=sys.stderr)
    self.difficulty = self.unwrapped._env._config._scenario_cfg.right_team_difficulty

    # FootballEnvCore
    self.footballEnvCore = self.unwrapped._env
    assert(self.footballEnvCore.__class__.__name__ == 'FootballEnvCore')

    # GameEnv
    self.gameEnv = self.footballEnvCore._env
    assert(self.gameEnv.__class__.__name__ == 'GameEnv')

    # Get scenario
    level = self.footballEnvCore._config['level']
    self.scenario = importlib.import_module(f'gfootball.scenarios.{level}')
    print(f'level={level} scenario={self.scenario}', file=sys.stderr)
    self.build_scenario = self.scenario.build_scenario

    self.raw_rewards = deque(maxlen=3)
    self.raw_reward = 0


  def step(self, action):
    observation, reward, done, info = self.env.step(action)
    self.raw_reward += float(info['score_reward'])
    if done:
        self.raw_rewards.append(self.raw_reward)
        print(f"game_reward={self.raw_reward} avg_raw_reward={np.mean(self.raw_rewards)} {self.raw_rewards}", file=sys.stderr)
        if len(self.raw_rewards) == 3 and np.mean(self.raw_rewards) >= 1.1:
            self.difficulty += 0.001
            self.raw_rewards = deque(maxlen=3)
    return observation, reward, done, info

  def reset(self):
    self.raw_reward = 0

    def build_scenario(builder):
      self.build_scenario(builder)
      builder.config().right_team_difficulty = self.difficulty

    self.scenario.build_scenario = build_scenario
    difficulty_prev = self.gameEnv.config.right_team_difficulty
    ret = self.env.reset()
    difficulty_current = self.gameEnv.config.right_team_difficulty
    print(f"[Reset] difficulty from {difficulty_prev} to {difficulty_current}", file=sys.stderr)
    return ret

def create_agent(unused_action_space, unused_env_observation_space,
                 parametric_action_distribution):
  return networks.GFootball(parametric_action_distribution)


def create_optimizer(unused_final_iteration):
  learning_rate_fn = lambda iteration: FLAGS.learning_rate
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  return optimizer, learning_rate_fn

def create_environment(_unused):
  e = env.create_environment(_unused)
  return DifficultyWrapper(e)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if FLAGS.run_mode == 'actor':
    actor.actor_loop(create_environment)
  elif FLAGS.run_mode == 'learner':
    learner.learner_loop(create_environment,
                         create_agent,
                         create_optimizer)
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  app.run(main)
