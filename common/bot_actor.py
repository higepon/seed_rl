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

"""SEED actor."""

import os
import timeit

from absl import flags
from absl import logging
import numpy as np
import psutil
import sys
from seed_rl import grpc
from seed_rl.common import common_flags
from seed_rl.common import profiling
from seed_rl.common import utils
from seed_rl.common import bot
import tensorflow as tf
import traceback
import collections


FLAGS = flags.FLAGS

# flags.DEFINE_integer('task', 0, 'Task id.')
# flags.DEFINE_integer('num_actors_with_summaries', 4,
#                      'Number of actors that will log debug/profiling TF '
#                      'summaries.')
# flags.DEFINE_bool('render', False,
#                   'Whether the first actor should render the environment.')


def are_summaries_enabled():
  return FLAGS.task < FLAGS.num_actors_with_summaries


def number_of_actors():
  num = 0
  for proc in psutil.process_iter():
      try:
          # Check if process name contains the given name string.
          # print("** proc name", proc.cmdline(), file=sys.stderr)
          if "--run_mode=actor" in proc.cmdline():
              num += 1
      except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
          pass
  return num

SMM_WIDTH = 96
SMM_HEIGHT = 72
# Normalized minimap coordinates
MINIMAP_NORM_X_MIN = -1.0
MINIMAP_NORM_X_MAX = 1.0
MINIMAP_NORM_Y_MIN = -1.0 / 2.25
MINIMAP_NORM_Y_MAX = 1.0 / 2.25
_MARKER_VALUE = 255
SMM_LAYERS = ['left_team', 'right_team', 'ball', 'active']

def get_smm_layers(config):
  return SMM_LAYERS

def mark_points(frame, points):
  """Draw dots corresponding to 'points'.
  Args:
    frame: 2-d matrix representing one SMM channel ([y, x])
    points: a list of (x, y) coordinates to be marked
  """
  for p in range(len(points) // 2):
    x = int((points[p * 2] - MINIMAP_NORM_X_MIN) /
            (MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN) * frame.shape[1])
    y = int((points[p * 2 + 1] - MINIMAP_NORM_Y_MIN) /
            (MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN) * frame.shape[0])
    x = max(0, min(frame.shape[1] - 1, x))
    y = max(0, min(frame.shape[0] - 1, y))
    frame[y, x] = _MARKER_VALUE

def generate_smm(observation, config=None,
                 channel_dimensions=(SMM_WIDTH, SMM_HEIGHT)):
  """Returns a list of minimap observations given the raw features for each
  active player.
  Args:
    observation: raw features from the environment
    config: environment config
    channel_dimensions: resolution of SMM to generate
  Returns:
    (N, H, W, C) - shaped np array representing SMM. N stands for the number of
    players we are controlling.
  """
  frame = np.zeros((len(observation), channel_dimensions[1],
                    channel_dimensions[0], len(get_smm_layers(config))),
                   dtype=np.uint8)

  for o_i, o in enumerate(observation):
    for index, layer in enumerate(get_smm_layers(config)):
      assert layer in o
      if layer == 'active':
        if o[layer] == -1:
          continue
        mark_points(frame[o_i, :, :, index],
                    np.array(o['left_team'][o[layer]]).reshape(-1))
      else:
        mark_points(frame[o_i, :, :, index], np.array(o[layer]).reshape(-1))
  return frame

def actor_loop(create_env_fn):
  """Main actor loop.

  Args:
    create_env_fn: Callable (taking the task ID as argument) that must return a
      newly created environment.
  """
  logging.info('Starting actor loop')
  print("*** num actors=", number_of_actors(), file=sys.stderr)
  is_rendering_enabled = FLAGS.render and FLAGS.task == 0
  if are_summaries_enabled():
    summary_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.logdir, 'actor_{}'.format(FLAGS.task)),
        flush_millis=20000, max_queue=1000)
    timer_cls = profiling.ExportingTimer
  else:
    summary_writer = tf.summary.create_noop_writer()
    timer_cls = utils.nullcontext

  actor_step = 0
  with summary_writer.as_default():
    while True:
      try:
        # Client to communicate with the learner.
        client = grpc.Client(FLAGS.server_address)
        env = create_env_fn(FLAGS.task)


        # Unique ID to identify a specific run of an actor.
        run_id = np.random.randint(np.iinfo(np.int64).max)
        observation, observation2 = env.reset()
        observations = collections.deque([], maxlen=4)
        reward = 0.0
        raw_reward = 0.0
        done = False
        abandoned = False
        global_step = 0
        episode_step = 0
        episode_step_sum = 0
        episode_return_sum = 0
        episode_raw_return_sum = 0
        episodes_in_report = 0

        elapsed_inference_s_timer = timer_cls('actor/elapsed_inference_s', 1000)
        last_log_time = timeit.default_timer()
        last_global_step = 0

        while True:

          tf.summary.experimental.set_step(actor_step)
          #print("***obs", observation, file=sys.stderr)

          # create SMM stacked
          #observation = observation['players_raw'][0]
          observation = generate_smm([observation])[0]
          if not observations:
              observations.extend([observation] * 4)
          else:
              observations.append(observation)
          observation = np.concatenate(list(observations), axis=-1)

          observation = np.concatenate(list(observations), axis=-1)
          observation = np.packbits(observation, axis=-1)
          if observation.shape[-1] % 2 == 1:
              observation = np.pad(observation, [(0, 0)] * (observation.ndim - 1) + [(0, 1)], 'constant')
          observation = observation.view(np.uint16)

          env_output = utils.EnvOutput(reward, done, observation,
                                       abandoned, episode_step)
          with elapsed_inference_s_timer:
            action = client.inference(
                FLAGS.task, run_id, env_output, raw_reward)

          # Get action of opponent bot.
          action2 = bot.agent(observation2)

          with timer_cls('actor/elapsed_env_step_s', 1000):
            [observation, observation2], [reward, reward2], done, info = env.step([action.numpy(), action2[0]])
          if is_rendering_enabled:
            env.render()
          episode_step += 1
          episode_return_sum += reward
          raw_reward = float((info or {}).get('score_reward', reward))
          episode_raw_return_sum += raw_reward
          # If the info dict contains an entry abandoned=True and the
          # episode was ended (done=True), then we need to specially handle
          # the final transition as per the explanations below.
          abandoned = (info or {}).get('abandoned', False)
          assert done if abandoned else True
          if done:
            # If the episode was abandoned, we need to report the final
            # transition including the final observation as if the episode has
            # not terminated yet. This way, learning algorithms can use the
            # transition for learning.
            if abandoned:
              # We do not signal yet that the episode was abandoned. This will
              # happen for the transition from the terminal state to the
              # resetted state.
              env_output = utils.EnvOutput(reward, False, observation,
                                           False, episode_step)
              with elapsed_inference_s_timer:
                action = client.inference(
                    FLAGS.task, run_id, env_output, raw_reward)
              reward = 0.0
              raw_reward = 0.0

            # Periodically log statistics.
            current_time = timeit.default_timer()
            episode_step_sum += episode_step
            global_step += episode_step
            episodes_in_report += 1
            if current_time - last_log_time > 1:
              logging.info(
                  'Actor steps: %i, Return: %f Raw return: %f Episode steps: %f, Speed: %f steps/s',
                  global_step, episode_return_sum / episodes_in_report,
                  episode_raw_return_sum / episodes_in_report,
                  episode_step_sum / episodes_in_report,
                  (global_step - last_global_step) /
                  (current_time - last_log_time))
              last_global_step = global_step
              episode_return_sum = 0
              episode_raw_return_sum = 0
              episode_step_sum = 0
              episodes_in_report = 0
              last_log_time = current_time

            # temporaly disabled this due to GPU memory error: https://github.com/tensorflow/tensorboard/issues/2485
            # to tensorboard @kuto
            # we should probably assert env.
            if hasattr(env, 'difficulty'):
             tf.summary.scalar('actor/difficulty', env.difficulty)
            # TODO: Should probably make checkpoint reward as FLAG
            if hasattr(env, 'checkpoint_reward'):
             tf.summary.scalar('actor/checkpoint reward', env.checkpoint_reward)
            summary_writer.flush()

            # Finally, we reset the episode which will report the transition
            # from the terminal state to the resetted state in the next loop
            # iteration (with zero rewards).
            with timer_cls('actor/elapsed_env_reset_s', 10):
              observation, observation2 = env.reset()
              episode_step = 0
            if is_rendering_enabled:
              env.render()
          actor_step += 1
      except (tf.errors.UnavailableError, tf.errors.CancelledError) as e:
        print(e, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        logging.exception(e)
        env.close()
