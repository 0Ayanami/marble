# MARBLE MultiAgentBench Experiments

`marble.mab_ex` is the configuration-driven experiment runner for running
MultiAgentBench-style tasks on MARBLE. The runner keeps system settings separate
from benchmark and dataset settings so new benchmarks can be added without
rewriting the experiment flow.

## Run

Use the MARBLE conda environment:

```powershell
conda activate marble
python -m marble.mab_ex.cli --config marble/configs/experiment_configs/mab_groupchat.yaml
```

System-level configs live in:

```text
marble/configs/experiment_configs/
```

The standard MARBLE/MultiAgentBench mode configs are:

- `mab_groupchat.yaml`
- `mab_groupchat_consensus.yaml`
- `mab_role_based.yaml`
- `mab_role_based_consensus.yaml`

The four modes are not implemented as four separate workflows. They are selected
by composition:

```yaml
collaboration:
  mode: groupchat   # or role_based

consensus:
  enabled: true     # or false
```

## Config Layout

System configs describe MARBLE runtime and consensus-layer behavior:

- `experiment`: experiment name, output root, run id, seed, and log level.
- `agents`: total agent count, default model, honest/byzantine model groups,
  model capability coefficients, byzantine count, and optional per-agent
  overrides.
- `collaboration`: MARBLE workflow selector. Use `groupchat` for
  Multi-Agent Discussion style runs and `role_based` for role collaboration.
- `consensus`: system-level consensus switch. `enabled: false` disables the
  proposal/verification/decision layer while keeping the same collaboration
  workflow.
- `proposal`: proposal builder, verification engine, policy, weight manager,
  and Fisher-IDA reserved parameters. Algorithm settings stay here; benchmark
  data does not.
- `memory`: shared-memory injection controls.
- `benchmark`: only benchmark index fields:
  - `type`: adapter name, for example `multiagentbench`.
  - `config_path`: path to benchmark-specific YAML.
  - `output_dir`: output directory for this run group.
- `sweep`: simple system-level sweep fields, currently `agent_counts` and
  `byzantine_counts`.

Benchmark-specific configs live in:

```text
marble/configs/experiment_configs/benchmarks/marble_configs/
```

Current MARBLE benchmark configs:

- `research.yaml`
- `minecraft.yaml`
- `database.yaml`
- `coding_challenge.yaml`

These files contain dataset paths, task selection, benchmark defaults, evaluator
options, and metric declarations. Different benchmarks may use different schemas;
the system parser loads the YAML as a dict and passes it to the adapter.

Example:

```yaml
benchmark:
  type: multiagentbench
  config_path: benchmarks/marble_configs/research.yaml
  output_dir: experiments/groupchat_consensus
```

## Metrics

Metrics are declared by benchmark config, not hard-coded in the system config.
The MultiAgentBench adapter currently emits MARBLE-compatible effectiveness and
efficiency fields, including KPI for the Research environment when
`options.evaluate_kpi: true`.

Common metric outputs include:

- effectiveness: KPI or task success signals.
- efficiency: task completion time, extra consensus time, interaction count,
  message count, extra message count, QC evolvement, and voting-agent-count
  evolvement.

## Results

Outputs are written under each config's `benchmark.output_dir`, grouped by
sweep and run id, for example:

```text
experiments/
  groupchat_consensus/
    agent5/
      byz0/
        groupchat_consensus_agent5_byz0/
```

Each task run directory contains:

- `config.yaml`: resolved system config plus loaded benchmark snapshot.
- `proposal_result.jsonl`: proposal, verification, decision, and consensus
  trace records.
- `benchmark_result.json`: benchmark and evaluator result payload.
- `metrics.json`: standardized metric payload.
- `summary.json`: compact run summary.
- `logs/`: MARBLE runtime logs.

Proposal traces and benchmark results are saved separately so consensus behavior
can be analyzed without mixing it into task-level benchmark scoring.
