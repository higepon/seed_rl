#!/bin/bash
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


set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/setup.sh

export CONFIG=football
export ENVIRONMENT=football
export AGENT=vtrace
# See https://docs.google.com/spreadsheets/d/12emT_Zc1Ckbp3gZDBL-0hktmnhLefyCNCXBx7Xye1-o/edit#gid=0
export WORKERS=4
export NUM_VCPU=96
export ACTORS_PER_WORKER=192


cat > /tmp/config.yaml <<EOF
trainingInput:
  scaleTier: CUSTOM
  masterType: n1-standard-4 # master is not cpu or memory bound.
  masterConfig:
    imageUri: ${IMAGE_URI}:${CONFIG}
    acceleratorConfig:
      count: 1
      type: NVIDIA_TESLA_P100 # TODO: Switch to better one NVIDIA_TESLA_P100, NVIDIA_TESLA_V100.
  workerCount: ${WORKERS}
  workerType: n1-standard-${NUM_VCPU}
  workerConfig:
    imageUri: ${IMAGE_URI}:${CONFIG}
  parameterServerCount: 0
  hyperparameters:
    goal: MAXIMIZE
    hyperparameterMetricTag: episode_return
    maxTrials: 1
    maxParallelTrials: 1
    enableTrialEarlyStopping: True
    params:
    - parameterName: game
      type: CATEGORICAL
      categoricalValues:
      - 11_vs_11_hard_stochastic
    - parameterName: init_checkpoint
      type: CATEGORICAL
      categoricalValues:
      - gs://higepon-kaggle-football-aiplatform/SEED_20201115163028/1/ckpt-1
    - parameterName: reward_experiment
      type: CATEGORICAL
      categoricalValues:
      - scoring,checkpoints
#    - parameterName: actors_per_worker
#      type: INTEGER
#      minValue: ${ACTORS_PER_WORKER}
#      maxValue: ${ACTORS_PER_WORKER}
#      scaleType: UNIT_LOG_SCALE
# higepon: We use default value -1 so that it can be autotuned.
#    - parameterName: inference_batch_size
#      type: INTEGER
#      minValue: 1
#      maxValue: 1
#      scaleType: UNIT_LOG_SCALE
    - parameterName: batch_size
      type: INTEGER
      minValue: 128
      maxValue: 128
      scaleType: UNIT_LOG_SCALE
# Unroll length 32 is best per their paper
    - parameterName: unroll_length
      type: INTEGER
      minValue: 32
      maxValue: 32
      scaleType: UNIT_LOG_SCALE
    - parameterName: total_environment_frames
      type: INTEGER
      minValue: 500000000
      maxValue: 500000000
      scaleType: UNIT_LOG_SCALE
    - parameterName: discounting
      type: DOUBLE
      minValue: 0.993
      maxValue: 0.993
      scaleType: UNIT_LOG_SCALE
    - parameterName: entropy_cost
      type: DOUBLE
      minValue: 0.0007330944745454107
      maxValue: 0.0007330944745454107
      scaleType: UNIT_LOG_SCALE
    - parameterName: lambda_
      type: DOUBLE
      minValue: 1
      maxValue: 1
      scaleType: UNIT_LOG_SCALE
    - parameterName: learning_rate
      type: DOUBLE
      minValue: 0.00019896
      maxValue: 0.00019896
      scaleType: UNIT_LOG_SCALE
EOF

start_training
