"""Read-only CPU audit of sparse-campaign data, parameters, panels, and budgets.

Only the output audit JSON is written. An incomplete campaign can be inspected
with --allow-incomplete, but is always marked incomplete and never passed.
Checkpoints are trusted local artifacts and are explicitly loaded onto CPU.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch

from data import Task
from sparse_data import tensor_hash


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


def frozen_hash(state, stage):
    """Match SparseClauseTransformer.frozen_state_hash without constructing it."""
    digest = hashlib.sha256()
    for name, parameter in state.items():
        frozen = name.startswith("ff.") if stage == "attention" else name in {"Q43", "Q44"}
        if frozen:
            value = parameter.detach().cpu().contiguous()
            digest.update(f"{name}:{tuple(value.shape)}:{value.dtype}".encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class Audit:
    def __init__(self):
        self.check_count = 0
        self.failed = []
        self.skipped = []
        self.per_run = {}

    def check(self, condition, check, subject, **detail):
        self.check_count += 1
        if not condition:
            self.failed.append({"check": check, "subject": subject, **canonical(detail)})
        return bool(condition)


def audit_campaign(root):
    root = Path(root).resolve()
    plan_path = root / "campaign-plan.json"
    plan = json.loads(plan_path.read_text())
    audit = Audit()
    specs = plan["runs"]
    names = [spec["name"] for spec in specs]
    audit.check(len(names) == len(set(names)), "unique_plan_names", "campaign-plan")
    audit.check(plan.get("total") == len(specs), "plan_count", "campaign-plan",
                declared=plan.get("total"), actual=len(specs))
    missing, checked = [], []
    configs, completions, data_records = {}, {}, {}
    checkpoint_cache = {}
    eval_hashes = defaultdict(dict)
    noise_data = {}

    def load_checkpoint(path, cache=False):
        path = Path(path).resolve()
        if path not in checkpoint_cache:
            value = torch.load(path, map_location="cpu", weights_only=False)
            if cache:
                checkpoint_cache[path] = value
            return value
        return checkpoint_cache[path]

    for spec in specs:
        name = spec["name"]
        directory = root / name
        complete_path = directory / "complete.json"
        if not complete_path.exists():
            missing.append(name)
            continue
        start_checks, start_failures = audit.check_count, len(audit.failed)
        try:
            complete = json.loads(complete_path.read_text())
            if not complete.get("complete"):
                missing.append(name)
                continue
            config = json.loads((directory / "config.json").read_text())
            result = json.loads((directory / "evaluation.json").read_text())
            data_audit = json.loads((directory / "data-audit.json").read_text())
            checkpoint_path = directory / "checkpoint-final.pt"
            checkpoint = load_checkpoint(checkpoint_path)
            initial = load_checkpoint(directory / "checkpoint-initial.pt")
            stage = config["stage"]
            model_state = checkpoint["model"]
            initial_state = initial["model"]
            configs[name], completions[name] = config, complete
            task = Task(config["task"], config["nvars"])

            audit.check(checkpoint["stage"] == stage == complete["stage"],
                        "stage_metadata", name)
            audit.check(checkpoint["model_config"] == config["model"],
                        "model_config_matches_checkpoint", name)
            audit.check(checkpoint["model_config"]["vocab_size"] == task.vocab_size
                        and checkpoint["model_config"]["nvars"] == task.nvars,
                        "task_vocabulary_matches_checkpoint", name)
            for key, value in canonical(checkpoint["args"]).items():
                audit.check(key in config and config[key] == value,
                            "checkpoint_arg_matches_config", name,
                            key=key, checkpoint=value, config=config.get(key))
            audit.check(checkpoint["step"] == complete["steps"],
                        "checkpoint_step_matches_completion", name)
            audit.check(Path(complete["checkpoint"]).resolve() == checkpoint_path.resolve(),
                        "completion_checkpoint_path", name)
            audit.check(complete["frozen_sha256_before"] == complete["frozen_sha256_after"]
                        == config["frozen_sha256_before"], "recorded_frozen_hashes", name)
            initial_frozen = frozen_hash(initial_state, stage)
            actual_frozen = frozen_hash(model_state, stage)
            audit.check(initial_frozen == complete["frozen_sha256_before"],
                        "actual_initial_frozen_hash", name,
                        actual=initial_frozen, recorded=complete["frozen_sha256_before"])
            audit.check(actual_frozen == complete["frozen_sha256_after"],
                        "actual_final_frozen_hash", name,
                        actual=actual_frozen, recorded=complete["frozen_sha256_after"])
            audit.check(result.get("complete") is True, "evaluation_completed", name)
            audit.check(sha256(checkpoint_path) == result["checkpoint_sha256"],
                        "evaluation_checkpoint_sha256", name)

            if stage == "attention":
                parent = load_checkpoint(config["init"], cache=True)
                audit.check(sha256(config["init"]) == config["init_sha256"],
                            "initialization_sha256", name)
                audit.check(parent["args"]["task"] == config["task"],
                            "initialization_task", name)
                audit.check(set(initial_state) == set(parent["model"]),
                            "initialization_parameter_names", name)
                for key, value in initial_state.items():
                    audit.check(key in parent["model"] and torch.equal(value, parent["model"][key]),
                                "initialization_parameter_exact", name, parameter=key)
                frozen_names = [key for key in model_state if key.startswith("ff.")]
                audit.check(bool(frozen_names), "ff_parameters_present", name)
                for key in frozen_names:
                    audit.check(key in initial_state and torch.equal(model_state[key], initial_state[key])
                                and key in parent["model"] and torch.equal(model_state[key], parent["model"][key]),
                                "ff_tensor_bitwise_frozen", name, parameter=key)
                if config.get("teacher"):
                    teacher = load_checkpoint(config["teacher"], cache=True)
                    audit.check(sha256(config["teacher"]) == config["teacher_sha256"],
                                "teacher_sha256", name)
                    audit.check(teacher["args"]["task"] == config["task"]
                                and teacher["model_config"]["nvars"] == task.nvars,
                                "teacher_task", name)
            else:
                audit.check(complete.get("local_gate_passed") is True,
                            "local_admission_gate", name)
                for key in ["Q43", "Q44"]:
                    audit.check(torch.equal(model_state[key], initial_state[key])
                                and torch.count_nonzero(model_state[key]).item() == 0,
                                "local_q_fixed_zero", name, parameter=key)

            training = torch.load(directory / "training-data.pt", map_location="cpu", weights_only=False)
            sample = training["sample"] if stage == "attention" else training
            source_hash = tensor_hash(sample["clauses"])
            recorded_source = data_audit.get("source_clause_sha256", data_audit.get("source_sha256"))
            audit.check(source_hash == recorded_source, "training_source_hash", name)
            audit.check(len(sample["clauses"]) == config["samples"], "training_example_count", name)
            if stage == "attention":
                supervision = training["supervision"]
                for field in ["inputs", "targets", "mask"]:
                    audit.check(tensor_hash(supervision[field]) == data_audit[field + "_sha256"],
                                "supervision_tensor_hash", name, field=field)
                labels_per_example = supervision["selected_targets"].shape[1]
                labels = int(supervision["mask"].sum())
                audit.check(labels == data_audit["supervised_value_labels"] == complete["unique_labels"],
                            "unique_label_budget_matches_data", name)
                audit.check(labels == config["samples"] * labels_per_example,
                            "constant_labels_per_example", name)
                if config["noise_kind"] in {"prefix_inconsistent", "prefix_consistent", "target"}:
                    noise_data[name] = {key: supervision[key] for key in ["corrupted", "inputs"]}
            else:
                labels_per_example = 5
                audit.check(complete["unique_labels"] == 5 * config["samples"],
                            "local_label_budget", name)
            audit.check(complete["labels_presented"] == complete["examples_presented"] * labels_per_example,
                        "presented_label_accounting", name)
            # Account for a shorter final minibatch at each epoch boundary.
            epoch_steps = (config["samples"] + config["batch"] - 1) // config["batch"]
            epochs, remainder = divmod(complete["steps"], epoch_steps)
            expected_examples = epochs * config["samples"] + min(remainder * config["batch"], config["samples"])
            audit.check(complete["examples_presented"] == expected_examples,
                        "presented_example_accounting", name,
                        expected=expected_examples, actual=complete["examples_presented"])
            data_records[name] = {"source_hash": source_hash,
                                  "label_source_hash": data_audit.get("label_source_clause_sha256")}
            for row in result["evaluations"]:
                key = (result["panel_seed"], row["length"], row["teacher_forced"]["samples"])
                eval_hashes[key][name] = row["panel_sha256"]
            checked.append(name)
        except Exception as exc:
            audit.check(False, "run_artifacts_readable", name,
                        exception=type(exc).__name__, error=str(exc))
        audit.per_run[name] = {"checks": audit.check_count - start_checks,
                               "failed_checks": len(audit.failed) - start_failures}

    # All length-four branches of a seed must share their exact parent model.
    init_groups = defaultdict(dict)
    source_groups = defaultdict(dict)
    for name in checked:
        c = configs[name]
        if c["stage"] == "attention" and c["length"] == 4:
            init_groups[c["seed"]][name] = c["init_sha256"]
        if name.startswith("extend_L4_"):
            key = (c["seed"], c["samples"], c["label_source"])
            source_groups[key][name] = data_records[name]
    for seed, values in init_groups.items():
        audit.check(len(set(values.values())) == 1, "shared_l4_initialization", f"seed={seed}", values=values)
    for key, values in source_groups.items():
        for field in ["source_hash", "label_source_hash"]:
            hashes = {name: row[field] for name, row in values.items()}
            audit.check(len(set(hashes.values())) == 1, "shared_supervision_branch_data",
                        str(key), field=field, values=hashes)
    for key, values in eval_hashes.items():
        audit.check(len(set(values.values())) == 1, "shared_evaluation_panel", str(key), values=values)

    for seed in plan.get("seeds", [0, 1, 2]):
        for source in ["truth", "teacher"]:
            all_name = f"labels_L4_N1024_{source}_all_s{seed}"
            for mode in ["second", "random"]:
                single_name = f"extend_L4_N4096_{source}_{mode}_s{seed}"
                pair = [all_name, single_name]
                if not all(name in checked for name in pair):
                    audit.skipped.append({"check": "equal_label_budget", "required_runs": pair,
                                          "reason": "required run incomplete or unreadable"})
                    continue
                for key in ["labels_presented", "unique_labels"]:
                    values = {name: completions[name][key] for name in pair}
                    audit.check(len(set(values.values())) == 1, "equal_label_budget", str(pair),
                                budget=key, values=values)
        for rate in [5, 20, 40]:
            group = [f"noise_L4_{kind}_r{rate:02d}_s{seed}" for kind in
                     ["target", "prefix_inconsistent", "prefix_consistent"]]
            if not all(name in noise_data for name in group):
                audit.skipped.append({"check": "paired_noise", "required_runs": group,
                                      "reason": "required run incomplete or unreadable"})
                continue
            target, inconsistent, consistent = group
            audit.check(torch.equal(noise_data[target]["corrupted"], noise_data[inconsistent]["corrupted"])
                        and torch.equal(noise_data[target]["corrupted"], noise_data[consistent]["corrupted"]),
                        "noise_corruption_masks_equal", str(group))
            audit.check(torch.equal(noise_data[inconsistent]["inputs"], noise_data[consistent]["inputs"]),
                        "noise_prefix_inputs_equal", str(group))

    campaign_complete = not missing and len(checked) == len(specs)
    passed = campaign_complete and not audit.failed and not audit.skipped
    status = "failed" if audit.failed else ("passed" if passed else "incomplete")
    return {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "campaign": str(root), "plan_sha256": sha256(plan_path),
            "audit_source_sha256": sha256(__file__), "status": status,
            "passed": passed, "campaign_complete": campaign_complete,
            "checks_passed_on_available_runs": not audit.failed,
            "planned_runs": len(specs), "checked_runs": len(checked),
            "missing_runs": missing, "checked_run_names": checked,
            "check_count": audit.check_count, "failed_checks": audit.failed,
            "skipped_checks": audit.skipped, "per_run": audit.per_run,
            "notes": ["CPU-only; source artifacts are never modified",
                      "Missing runs and skipped comparisons prevent an overall passed result",
                      "Frozen FFN tensors are compared bitwise with both saved initialization and parent checkpoint",
                      "Source-code hashes may differ across runs; each run's recorded provenance is retained"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "runs/sparse-supervision-2026-11-09")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = audit_campaign(args.root)
    output = args.output or args.root / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), **{key: result[key] for key in
          ["status", "passed", "planned_runs", "checked_runs", "check_count"]},
          "failed_checks": result["failed_checks"], "missing_runs": result["missing_runs"]}, ensure_ascii=False))
    if result["failed_checks"]:
        sys.exit(1)
    if not result["passed"] and not args.allow_incomplete:
        sys.exit(2)


if __name__ == "__main__":
    main()
