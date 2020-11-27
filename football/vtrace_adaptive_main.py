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
from seed_rl.common import actor, bot_actor, random_forest_bot_actor
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

# Custom settings by kuto and higpon.
flags.DEFINE_bool('adaptive_learning', True,
                  'Whether adjust difficulty as training goes.')
flags.DEFINE_float('initial_difficulty', 1.0, 'initial difficulty')

flags.DEFINE_bool('custom_checkpoints', False,
                  'Whether custom checkpoints rewward is enabled.')

# https://sites.google.com/view/rl-football/singleagent-team
class DifficultyWrapper(gym.Wrapper):
  def __init__(self, env, initial_difficulty, customCheckpointRewardWrapper):
    # Call the parent constructor, so we can access self.env later
    super(DifficultyWrapper, self).__init__(env)
    print(f"Initialized DifficultyWrapper {self.unwrapped._env._config._scenario_cfg.right_team_difficulty}", file=sys.stderr)
    self.difficulty = initial_difficulty

    # FootballEnvCore
    self.footballEnvCore = self.unwrapped._env
    assert(self.footballEnvCore.__class__.__name__ == 'FootballEnvCore')

    # GameEnv
    self.gameEnv = self.footballEnvCore._env
    assert(self.gameEnv.__class__.__name__ == 'GameEnv')

    # Checkpoint wrapper
    self.customCheckpointRewardWrapper = customCheckpointRewardWrapper

    # Get scenario
    level = self.footballEnvCore._config['level']
    self.scenario = importlib.import_module(f'gfootball.scenarios.{level}')
    print(f'level={level} scenario={self.scenario}', file=sys.stderr)
    self.build_scenario = self.scenario.build_scenario


  def step(self, action):
    observation, reward, done, info = self.env.step(action)
    return observation, reward, done, info

  def reset(self):
    def build_scenario(builder):
      self.build_scenario(builder)
      builder.config().right_team_difficulty = self.difficulty

    self.scenario.build_scenario = build_scenario
    difficulty_prev = self.gameEnv.config.right_team_difficulty
    ret = self.env.reset()
    difficulty_current = self.gameEnv.config.right_team_difficulty
    print(f"[Reset] difficulty from {difficulty_prev} to {difficulty_current}", file=sys.stderr)
    if self.customCheckpointRewardWrapper:
      self.customCheckpointRewardWrapper.checkpoint_reward = 0.0

    return ret

# add custom reward wrapper @kuto
class CustomCheckpointRewardWrapper(gym.RewardWrapper):
  """A wrapper that adds a dense checkpoint reward."""

  def __init__(self, env):
    gym.RewardWrapper.__init__(self, env)
    self._collected_checkpoints = {}
    self._num_checkpoints = 10
    self.checkpoint_reward = 0.0

  def reset(self):
    self._collected_checkpoints = {}
    return self.env.reset()

  def get_state(self, to_pickle):
    to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
    return self.env.get_state(to_pickle)

  def set_state(self, state):
    from_pickle = self.env.set_state(state)
    self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
    return from_pickle

  def reward(self, reward):
    reward = [reward]
    observation = self.env.unwrapped.observation()
    if observation is None:
      return reward

    assert len(reward) == len(observation)

    for rew_index in range(len(reward)):
      o = observation[rew_index]
      if reward[rew_index] == 1:
        reward[rew_index] += np.float32(self.checkpoint_reward * (
            self._num_checkpoints -
            self._collected_checkpoints.get(rew_index, 0)))
        self._collected_checkpoints[rew_index] = self._num_checkpoints
        continue

      # Check if the active player has the ball.
      if ('ball_owned_team' not in o or
          o['ball_owned_team'] != 0 or
          'ball_owned_player' not in o or
          o['ball_owned_player'] != o['active']):
        continue

      d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5

      # Collect the checkpoints.
      # We give reward for distance 1 to 0.2.
      while (self._collected_checkpoints.get(rew_index, 0) <
             self._num_checkpoints):
        if self._num_checkpoints == 1:
          threshold = 0.99 - 0.8
        else:
          threshold = (0.99 - 0.8 / (self._num_checkpoints - 1) *
                       self._collected_checkpoints.get(rew_index, 0))
        if d > threshold:
          break
        # Note: self.checkpoint_reward is python float(64bit).
        # But this reward is supposed to be np.float32.
        reward[rew_index] += np.float32(self.checkpoint_reward)
        self._collected_checkpoints[rew_index] = (
            self._collected_checkpoints.get(rew_index, 0) + 1)
    return reward[0]


def create_agent(unused_action_space, unused_env_observation_space,
                 parametric_action_distribution):
  return networks.GFootball(parametric_action_distribution)


def create_optimizer(unused_final_iteration):
  learning_rate_fn = lambda iteration: FLAGS.learning_rate
  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  return optimizer, learning_rate_fn

def create_environment(_unused):
  if FLAGS.run_mode == 'bot_actor':
    e = env.create_environment_for_actor(_unused)
  else:
    e = env.create_environment(_unused)
  print("**** Adaptive {}({}) Custom checkpoint {}".format(FLAGS.adaptive_learning, FLAGS.initial_difficulty, FLAGS.custom_checkpoints), file=sys.stderr)
  if FLAGS.custom_checkpoints:
    print("**** Custom checkpoints reward enabled ****", file=sys.stderr)
    e = CustomCheckpointRewardWrapper(e)  # add @kuto
  if FLAGS.adaptive_learning:
    print("**** Adaptive learning enabled ****", file=sys.stderr)
    e = DifficultyWrapper(e, FLAGS.initial_difficulty, e if FLAGS.custom_checkpoints else None)
  return e

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if FLAGS.run_mode == 'actor':
    print("*** Starting normal actors", file=sys.stderr)
    actor.actor_loop(create_environment)
  elif FLAGS.run_mode == 'bot_actor':
    print("*** Starting bot actor", file=sys.stderr)
    bot_actor.actor_loop(create_environment)
  elif FLAGS.run_mode == 'random_forest_bot_actor':
    print("*** Starting random forest bot actor", file=sys.stderr)
    random_forest_bot_actor.actor_loop(create_environment)
  elif FLAGS.run_mode == 'learner':
    learner.learner_loop(create_environment,
                         create_agent,
                         create_optimizer)
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  app.run(main)
