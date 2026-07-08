# MARBLE MultiAgentBench Experiments

`marble.mab_ex` is the MultiAgentBench-specific experiment runner for MARBLE.
It is configuration-driven and currently validates the first-stage Research
Environment experiment. The same runner is prepared for Minecraft, Database,
and Coding Challenge through environment adapters.

## Run

Use the MARBLE conda environment:

```powershell
conda activate marble
python -m marble.mab_ex.cli --config marble/configs/marble_configs/mab_groupchat.yaml
```

The four first-stage configs are:

- `marble/configs/marble_configs/mab_groupchat.yaml`
- `marble/configs/marble_configs/mab_role_based.yaml`
- `marble/configs/marble_configs/mab_groupchat_consensus.yaml`
- `marble/configs/marble_configs/mab_role_based_consensus.yaml`

Each config runs Research `task_id: 1-10` and uses `sweep.agent_counts: [5, 7]`,
so one command produces the 5-agent and 7-agent groups for that workflow.

## Configuration

All MAB configs live under:

```text
marble/configs/marble_configs/
```

Important sections:

- `experiment`: experiment name, root output directory, and base run id.
- `agents`: agent count, model, capability coefficient, optional profiles, and
  per-agent overrides. For this stage use `model: qwen3.7-max-preview`,
  `capability_coefficient: 8.96`, and `num_byzantine: 0`.
- `collaboration`: workflow selector. Use `groupchat` for SelectorGroupChat/MAD
  style discussion and `role_based` for Magentic-One-style role collaboration.
- `proposal`: consensus/proposal layer. Set `enabled: false` for non-consensus
  baselines. When enabled, use `consensus.strategy: smart_quorum`,
  `verification.type: llm`, and `fisher_ida.enabled: false`.
- `benchmark`: MultiAgentBench dataset, environment name, task selection, output
  directory, and evaluator options.
- `sweep`: experiment expansion. `agent_counts: [5, 7]` runs both requested
  agent scales with the same task/workflow/consensus settings.

Environment placeholder configs are stored in:

```text
marble/configs/marble_configs/environments/
```

Currently `research.yaml` is runnable. `minecraft.yaml`, `database.yaml`, and
`coding_challenge.yaml` reserve the initialization shape for later phases.

## Results

All outputs are written under `experiments/`, grouped by workflow and agent
count, for example:

```text
experiments/
  groupchat/
    agent5/
    agent7/
  role_based/
    agent5/
    agent7/
  groupchat_consensus/
    agent5/
    agent7/
  role_based_consensus/
    agent5/
    agent7/
```

Each task run directory contains at least:

- `config.yaml`: exact resolved run config.
- `metrics.json`: standardized metric payload.
- `summary.json`: compact task summary.
- `logs/`: MARBLE runtime logs.

Additional detailed artifacts:

- `benchmark_result.json`: raw benchmark/evaluator result.
- `proposal_result.jsonl`: proposal, verification, and consensus records.

Each sweep directory also writes:

- `metrics.json`: group-level average metrics.
- `summary.json`: group summary and per-task result pointers.
- `sweep_progress.jsonl`: append-only progress log.

The standardized metrics include:

- `task_success_rate`
- `overall_score`
- `average_task_score`
- `completion_rate`
- `task_scores`
- row-style records with `task_id`, `environment`, `workflow`, `consensus`,
  `agent_number`, `metric`, and `score`.

The repository does not currently include a separate MultiAgentBench official
Research evaluator implementation. The adapter therefore normalizes MARBLE
Evaluator completion/KPI signals into the required metric schema and records
`evaluator_source: marble_evaluator_fallback`.
