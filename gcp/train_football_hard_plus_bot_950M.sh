## 最初の training 用
##
## - steps: 500M
## - reward: scoring + custom checkpoints.
## - difficulty: easy to 1.0 using adaptive
## - GPU: NVIDIA_TESLA_P100
## - CPU: n1-standard-96 * 4
## - Other hyper parameters are from the paper.


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
    - parameterName: reward_experiment
      type: CATEGORICAL
      categoricalValues:
      - scoring
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
    - parameterName: init_checkpoint
      type: CATEGORICAL
      categoricalValues:
      # - gs://higepon-kaggle-football-aiplatform/SEED_hard_plus_bot_1650M/1/ckpt-257
      - gs://oceanic-hook-237214-aiplatform/SEED_hard_plus_bot_1600M/1/ckpt-247
    - parameterName: total_environment_frames
      type: INTEGER
      minValue: 1620000000
      maxValue: 1620000000
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
    - parameterName: initial_difficulty
      type: DOUBLE
      minValue: 1.0 # hard
      maxValue: 1.0 # hard
      scaleType: UNIT_LOG_SCALE
EOF

start_training
