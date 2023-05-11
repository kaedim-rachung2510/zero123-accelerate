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
pin_and_add_path "taming-transformers" "3ba01b2"
pin_and_add_path "CLIP" "a9b1bf5"
pin_and_add_path "image-background-remove-tool" "2935e46"

# A single requirements file for all the submodules
pip install -r requirements.txt

# Patching zero123's ldm source to work with bumped lightning - don't fork or vendor for a single line of change
(cd ./zero123/zero123/ldm/models/diffusion && patch < ../../../../../patches/ldm.patch)
