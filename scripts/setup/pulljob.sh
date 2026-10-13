#!/bin/sh
#SBATCH -p cbuild
#SBATCH -t 00:20:00
# Pull the dependency image into ./pidu-env.sif (the checkout is bind-mounted at run time).
export APPTAINER_TMPDIR="$(mktemp -d)"
apptainer pull pidu-env.sif docker://ghcr.io/mkoumans/pidu-env:latest
