# Cluster workflow

- The cluster requires Slurm for running scripts and tests. Do not tell the user to run `python`, `bash`, or training commands directly on a login node.
- Before upload, run `py scripts/preflight_upload.py` locally. Use `--fix` locally if it reports CRLF line endings.
- Upload with `bash sync.sh`, then run `bash sync.sh --verify` locally. Verification uses rsync checksums and does not run project scripts on the cluster.
- Submit training with `sbatch nanochat_train.sh`. The job runs preflight inside the allocation before training.
- `moduleAvail.md` lists `shared`, `python3/anaconda/3.12`, and `cuda/12.8`, matching the modules in `nanochat_train.sh`. Recheck the list if the cluster configuration changes.
