# Cluster workflow

- The cluster requires Slurm for running scripts and tests. Do not tell the user to run `python`, `bash`, or training commands directly on a login node.
- Before upload, run `py scripts/preflight_upload.py` locally. Use `--fix` locally if it reports CRLF line endings.
- Upload with `bash sync.sh`, then run `bash sync.sh --verify` locally. Verification uses rsync checksums and does not run project scripts on the cluster.
- Submit training with `sbatch nanochat_train.sh`. The job runs preflight inside the allocation before training.
- Do not assume a module exists from its name. Check `moduleAvail.md` if populated, or use a Slurm job to query modules. The current `moduleAvail.md` is empty.
