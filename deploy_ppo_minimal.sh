#!/bin/bash

# Minimal deployment script for PPO policy
# Only copies necessary files instead of entire workspace

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  REMOTE=`avahi-resolve-host-name raspberrypi.local -4 | awk '{print $2}'`
elif [[ "$OSTYPE" == "darwin"* ]]; then
  REMOTE=192.168.1.100 #raspberrypi.local
else
  print "not supported os"
  exit 1
fi

echo "Robot address: ${REMOTE}"
REMOTE_DIR=/home/pi/puppersim_ppo

# Create remote directory
ssh pi@${REMOTE} mkdir -p ${REMOTE_DIR}

# Only copy essential files
echo "Copying essential files..."

# Copy the deployment script
rsync -avz puppersim/pupper_deploy_ppo.py pi@${REMOTE}:${REMOTE_DIR}/

# Copy the model file (extract from command line args)
MODEL_PATH=""
for arg in "$@"; do
    if [[ $arg == --model_path=* ]]; then
        MODEL_PATH="${arg#*=}"
    elif [[ $prev_arg == "--model_path" ]]; then
        MODEL_PATH="$arg"
    fi
    prev_arg="$arg"
done

if [[ -n "$MODEL_PATH" ]]; then
    echo "Copying model: $MODEL_PATH"
    rsync -avz "$MODEL_PATH" pi@${REMOTE}:${REMOTE_DIR}/model.cleanrl_model
    # Update the command to use the copied model
    NEW_ARGS="${@//$MODEL_PATH/model.cleanrl_model}"
else
    NEW_ARGS="$@"
fi

# Copy core puppersim modules (only what's needed)
rsync -avz puppersim/__init__.py pi@${REMOTE}:${REMOTE_DIR}/puppersim/
rsync -avz puppersim/pupper_train_ppo_cont_action.py pi@${REMOTE}:${REMOTE_DIR}/puppersim/
rsync -avz puppersim/pupper_gym_env.py pi@${REMOTE}:${REMOTE_DIR}/puppersim/
rsync -avz puppersim/config/ pi@${REMOTE}:${REMOTE_DIR}/puppersim/
rsync -avz puppersim/data/ pi@${REMOTE}:${REMOTE_DIR}/puppersim/

# Run the command
if [ -z "$1" ] ; then
  ssh -t pi@${REMOTE} "cd ${REMOTE_DIR} ; bash --login"
else
  echo "Running: $NEW_ARGS"
  ssh -t pi@${REMOTE} "cd ${REMOTE_DIR} ; $NEW_ARGS"
fi
