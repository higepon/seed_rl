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
export WORKERS=1
export ACTORS_PER_WORKER=48

cat > /tmp/config.yaml <<EOF
trainingInput:
  scaleTier: CUSTOM
  masterType: n1-standard-4 # master is not cpu or memory bound.
  masterConfig:
    imageUri: ${IMAGE_URI}:${CONFIG}
    acceleratorConfig:
      count: 1
      type: NVIDIA_TESLA_K80 # TODO: Switch to better one NVIDIA_TESLA_P100, NVIDIA_TESLA_V100.
  workerCount: ${WORKERS}
  workerType: n1-highmem-8
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
    - parameterName: reward_experiment
      type: CATEGORICAL
      categoricalValues:
      - scoring,checkpoints
#    - parameterName: actors_per_worker
#      type: INTEGER
#      minValue: ${ACTORS_PER_WORKER}
#      maxValue: ${ACTORS_PER_WORKER}
#      scaleType: UNIT_LOG_SCALE
    - parameterName: inference_batch_size
      type: INTEGER
      minValue: 1
      maxValue: 1
      scaleType: UNIT_LOG_SCALE
    - parameterName: batch_size
      type: INTEGER
      minValue: 64
      maxValue: 64
      scaleType: UNIT_LOG_SCALE
    - parameterName: unroll_length
      type: INTEGER
      minValue: 32
      maxValue: 32
      scaleType: UNIT_LOG_SCALE
    - parameterName: total_environment_frames
      type: INTEGER
      minValue: 150000
      maxValue: 150000
      scaleType: UNIT_LOG_SCALE
    - parameterName: discounting
      type: DOUBLE
      minValue: 0.997
      maxValue: 0.997
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
      minValue: 0.00012542101122072784
      maxValue: 0.00012542101122072784
      scaleType: UNIT_LOG_SCALE
EOF

start_training
