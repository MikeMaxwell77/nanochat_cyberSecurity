# CTF data control run

This control starts from the same base checkpoint as a completed CTF fine-tuning run, trains for the same number of optimizer steps without CTF examples, and evaluates on that run's saved cyber question suite. The comparison includes general task scores.

First upload the new files from MobaXterm:

```bash
bash sync.sh --diff
bash sync.sh
```

Wait for the treatment job to complete successfully. For job 805543:

```bash
sacct -j 805543 --format=JobID,State,ExitCode
```

Find its fine-tuned checkpoint tag under `~/.cache/nanochat/chatsft_checkpoints/`. It has the form `<base-tag>-cyber-<timestamp>`; match it to the completed job's output. Submit the control with that exact directory name:

```bash
sbatch --export=ALL,TREATMENT_TAG=<checkpoint-tag> nanochat_ctf_control.sh
```

The job refuses to start without a final treatment checkpoint and its saved evaluation suite. It saves a separate `<checkpoint-tag>-no-ctf` model and writes `ctf_ablation.md` under `~/.cache/nanochat/comparisons/<checkpoint-tag>-no-ctf/`. A failed or incomplete control can leave this output tag behind; move that tag aside before retrying.

The control matches optimizer steps, base checkpoint, data mixture outside CTF, and evaluation questions. The original treatment used dataset-driven progress while the control uses fixed-step progress, so their learning-rate schedules may differ slightly. Treat small score differences cautiously.