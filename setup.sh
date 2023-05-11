#!/bin/bash

pin_and_add_path() {
  local module="$1"
  local commit_sha="$2"
  local module_path="$(cd "$module" && git reset --hard "$commit_sha" && realpath . 2>&1 | tail -n 1)"
  if [ $? -eq 0 ]; then
    export PYTHONPATH="${PYTHONPATH}:${module_path}"
    printf "Added %s with commit %s to PYTHONPATH\n" "$module" "$commit_sha"
  else
    printf "Error: %s\n" "module_path"
  fi
}

# Add submodules to PYTHONPATH. Pin to specific commits.
pin_and_add_path "zero123" "78bc429"
pin_and_add_path "taming-transformers" "3ba01b241669f5ade541ce990f7650a3b8f65318"
pin_and_add_path "CLIP" "a9b1bf5920416aaeaec965c25dd9e8f98c864f16"
pin_and_add_path "image-background-remove-tool" "2935e4655d2c0260195e22ac08af6c43b5969fdd"

# A single requirements file for all the submodules
pip install -r requirements.txt

# Patching zero123's ldm source to work with bumped lightning - don't fork or vendor for a single line of change
(cd ./zero123/zero123/ldm/models/diffusion && patch < ../../../../../patches/ldm.patch)
