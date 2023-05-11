#!/bin/bash

pin_and_add_path() {
  local dir="$1"
  local revision="$2"
  local output=$(cd "$dir" && git reset --hard "$revision" && realpath . 2>&1 | tail -n 1)
  if [ $? -eq 0 ]; then
    export PYTHONPATH="${PYTHONPATH}:${output}"
    echo "Module directory: ${output}"
  else
    echo "Error: ${output}"
  fi
}

# Add submodules to PYTHONPATH. Pin commits.
pin_and_add_path "zero123" "78bc42957960e0c0396e7866e7c03b1f6e72dcfe"
pin_and_add_path "taming-transformers" "3ba01b241669f5ade541ce990f7650a3b8f65318"
pin_and_add_path "CLIP" "a9b1bf5920416aaeaec965c25dd9e8f98c864f16"
pin_and_add_path "image-background-remove-tool" "2935e4655d2c0260195e22ac08af6c43b5969fdd"

# A single requirements file for all the submodules
pip install -r requirements.txt

# Patching zero123's ldm source, to allow bumping pytorch-lightning - don't fork or vendor for a single line of change
!patch < ./patches/ldm.patch
