#!/usr/bin/env python3
"""
validate_augmented.py — Standalone validator for the output of augmentation.py.

Runs five independent checks against the augmented dataset and the MUSAN corpus:

    V1  Audio actually changed     (sampled waveform comparison vs source)
    V3  Status distribution        (counts per aug_status value)
    V4  Schema consistency         (audio dict shape, types)
    V5  Full decode test           (every row's audio.bytes/path is decodable)
    V7  MUSAN preflight            (required subdirs exist with N+ files)

Exit code is 0 if every check passes, 1 otherwise. Designed to be wired into
CI or a pre-Section-7 gate.

Usage:
    python validate_augmented.py \
        --output-ds birlashtirilgan_dataset_augmented \
        --input-audio-dir ./audio_segments \
        --musan-path ./musan
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ─────────────────────────────────────────
# Result type
# ─────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    summary: str
    metrics: dict = field(default_factory=dict)

    def report(self) -> str:
        tag = "[PASS]" if self.passed else "[FAIL]"
        lines = [f"{tag} {self.name}: {self.summary}"]
        for k, v in self.metrics.items():
            lines.append(f"        {k}: {v}")
        return "\n".join(lines)


# ─────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────

PREFIX_AUGMENTED = "augmented_"
PREFIX_AUG = "aug_"


def _strip_prefix(basename: str) -> str:
    if basename.startswith(PREFIX_AUGMENTED):
        return basename[len(PREFIX_AUGMENTED):]
    if basename.startswith(PREFIX_AUG):
        return basename[len(PREFIX_AUG):]
    return basename


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _decode_audio_entry(entry: Any) -> tuple[np.ndarray, int]:
    """Return (samples, sr). Raises on failure."""
    import soundfile as sf
    if isinstance(entry, dict):
        b = entry.get("bytes")
        if b:
            data, sr = sf.read(io.BytesIO(b))
            return np.asarray(data), sr
        p = entry.get("path")
        if p:
            data, sr = sf.read(p)
            return np.asarray(data), sr
        raise ValueError("audio dict has neither bytes nor path")
    if isinstance(entry, str):
        data, sr = sf.read(entry)
        return np.asarray(data), sr
    raise TypeError(f"unsupported audio entry type: {type(entry).__name__}")


# ─────────────────────────────────────────
# V7 — MUSAN preflight
# ─────────────────────────────────────────

REQUIRED_MUSAN_SUBDIRS = (
    "noise/free-sound",
    "noise/sound-bible",
    "speech/librivox",
    "music/jamendo",
)


def check_musan_preflight(musan_path: str, min_files: int) -> CheckResult:
    if not os.path.isdir(musan_path):
        return CheckResult(
            name="V7 MUSAN preflight",
            passed=False,
            summary=f"MUSAN root does not exist: {musan_path}",
        )

    counts: dict[str, int] = {}
    failures: list[str] = []

    for sub in REQUIRED_MUSAN_SUBDIRS:
        full = os.path.join(musan_path, sub)
        n = 0
        if os.path.isdir(full):
            for root, _, files in os.walk(full):
                # Files moved by filter_musan_files() live under _quarantine — skip
                if "_quarantine" in root.replace("\\", "/").split("/"):
                    continue
                n += sum(1 for f in files if f.endswith(".wav"))
        counts[sub] = n
        if n < min_files:
            failures.append(f"{sub} has {n} (need ≥ {min_files})")

    if failures:
        return CheckResult(
            name="V7 MUSAN preflight",
            passed=False,
            summary=f"{len(failures)}/{len(REQUIRED_MUSAN_SUBDIRS)} subdirs below threshold",
            metrics={**counts, "failed_subdirs": "; ".join(failures)},
        )

    return CheckResult(
        name="V7 MUSAN preflight",
        passed=True,
        summary=f"all {len(REQUIRED_MUSAN_SUBDIRS)} subdirs present with ≥ {min_files} files",
        metrics=counts,
    )


# ─────────────────────────────────────────
# V3 — Status distribution (aug_status field)
# ─────────────────────────────────────────

ALLOWED_STATUSES = ("augmented", "skip_prob", "skip_long", "failed", "invalid_input")


def check_status_distribution(
    ds,
    aug_prob: float,
    tolerance: float,
    failed_ratio_threshold: float = 0.02,
    invalid_input_ratio_threshold: float = 0.01,
) -> CheckResult:
    """
    Count rows by aug_status and verify the distribution.

    PASS requires ALL of:
      - effective_total > 0   (effective_total = total_rows - invalid_input)
      - |actual_augmented_ratio - aug_prob| <= tolerance
      - failed_ratio        <= failed_ratio_threshold
      - invalid_input_ratio <= invalid_input_ratio_threshold
      - no rows with an unrecognized aug_status   (hard FAIL — no tolerance)

    Real-world pipelines may produce a small fraction of failed/invalid rows
    (transient I/O errors, occasional NaN from extreme transforms, broken
    sources from upstream). The two ratio thresholds let operators set a
    realistic floor for "healthy" without flipping to FAIL on a single
    flake.
    """
    total_rows = len(ds)
    counts = {s: 0 for s in ALLOWED_STATUSES}
    unknown = 0

    for row in ds:
        s = row.get("aug_status")
        if s in counts:
            counts[s] += 1
        else:
            unknown += 1

    effective_total = total_rows - counts["invalid_input"]
    actual_augmented_ratio = (
        counts["augmented"] / effective_total if effective_total > 0 else 0.0
    )
    diff = abs(actual_augmented_ratio - aug_prob)

    failed_ratio = counts["failed"] / total_rows if total_rows > 0 else 0.0
    invalid_input_ratio = (
        counts["invalid_input"] / total_rows if total_rows > 0 else 0.0
    )

    failures: list[str] = []
    if effective_total == 0:
        failures.append("effective_total is 0 (every row is invalid_input)")
    if effective_total > 0 and diff > tolerance:
        failures.append(
            f"augmented ratio {actual_augmented_ratio:.1%} deviates from "
            f"expected {aug_prob:.1%} by {diff:.1%} (tol {tolerance:.1%})"
        )
    if failed_ratio > failed_ratio_threshold:
        failures.append(
            f"failed_ratio {failed_ratio:.2%} exceeds "
            f"threshold {failed_ratio_threshold:.2%} "
            f"({counts['failed']} of {total_rows} rows)"
        )
    if invalid_input_ratio > invalid_input_ratio_threshold:
        failures.append(
            f"invalid_input_ratio {invalid_input_ratio:.2%} exceeds "
            f"threshold {invalid_input_ratio_threshold:.2%} "
            f"({counts['invalid_input']} of {total_rows} rows)"
        )
    if unknown > 0:
        # Unknown status is always a hard FAIL — it means the dataset was
        # produced by code outside our pipeline, so no threshold applies.
        failures.append(f"{unknown} rows with unrecognized aug_status (hard fail)")

    passed = not failures
    summary = (
        f"augmented={actual_augmented_ratio:.1%} (expected≈{aug_prob:.1%}, "
        f"diff {diff:.1%}, tol {tolerance:.1%}); "
        f"failed={failed_ratio:.2%} (max {failed_ratio_threshold:.2%}); "
        f"invalid_input={invalid_input_ratio:.2%} (max {invalid_input_ratio_threshold:.2%})"
    )
    if failures:
        summary += "; " + "; ".join(failures)

    return CheckResult(
        name="V3 Status distribution",
        passed=passed,
        summary=summary,
        metrics={
            "total": total_rows,
            "augmented": counts["augmented"],
            "skip_prob": counts["skip_prob"],
            "skip_long": counts["skip_long"],
            "failed": counts["failed"],
            "invalid_input": counts["invalid_input"],
            "augmented_ratio": round(actual_augmented_ratio, 4),
            "failed_ratio": round(failed_ratio, 4),
            "invalid_input_ratio": round(invalid_input_ratio, 4),
        },
    )


# ─────────────────────────────────────────
# V4 — Schema consistency
# ─────────────────────────────────────────

def check_schema(ds) -> CheckResult:
    expected_cols = {"audio", "text", "duration"}
    missing_cols = expected_cols - set(ds.column_names)
    if missing_cols:
        return CheckResult(
            name="V4 Schema",
            passed=False,
            summary=f"missing columns: {sorted(missing_cols)}",
            metrics={"columns": ds.column_names},
        )

    issues = {
        "audio_not_dict_or_str": 0,
        "audio_no_bytes_no_path": 0,
        "audio_path_only_no_bytes": 0,  # portability risk, soft warning
        "text_not_str": 0,
        "duration_not_number": 0,
    }
    n = len(ds)

    for row in ds:
        audio = row.get("audio")
        if isinstance(audio, dict):
            b, p = audio.get("bytes"), audio.get("path")
            if b is None and not p:
                issues["audio_no_bytes_no_path"] += 1
            elif b is None and p:
                issues["audio_path_only_no_bytes"] += 1
        elif isinstance(audio, str):
            pass  # treated as a path, OK
        else:
            issues["audio_not_dict_or_str"] += 1

        if not isinstance(row.get("text"), str):
            issues["text_not_str"] += 1
        if not isinstance(row.get("duration"), (int, float)):
            issues["duration_not_number"] += 1

    hard_failures = (
        issues["audio_not_dict_or_str"]
        + issues["audio_no_bytes_no_path"]
        + issues["text_not_str"]
        + issues["duration_not_number"]
    )

    return CheckResult(
        name="V4 Schema",
        passed=hard_failures == 0,
        summary=(
            f"{hard_failures} hard issues; "
            f"{issues['audio_path_only_no_bytes']} path-only rows (portability risk)"
        ),
        metrics={"total_rows": n, **issues},
    )


# ─────────────────────────────────────────
# V5 — Full decode test
# ─────────────────────────────────────────

def check_decode_all(ds, limit: Optional[int]) -> CheckResult:
    n = len(ds)
    iter_n = min(n, limit) if limit else n
    failures: list[tuple[int, str]] = []

    for i in range(iter_n):
        row = ds[i]
        try:
            samples, sr = _decode_audio_entry(row.get("audio"))
            if samples.size == 0:
                failures.append((i, "decoded array is empty"))
            elif sr <= 0:
                failures.append((i, f"invalid sample rate: {sr}"))
        except Exception as e:
            failures.append((i, f"{type(e).__name__}: {e}"))

    return CheckResult(
        name="V5 Decode all",
        passed=len(failures) == 0,
        summary=f"{len(failures)}/{iter_n} rows failed to decode",
        metrics={
            "scanned": iter_n,
            "failed": len(failures),
            "first_failures": failures[:5],
        },
    )


# ─────────────────────────────────────────
# V1 — Audio actually changed
# ─────────────────────────────────────────

def _looks_unmodified(in_samples: np.ndarray, out_samples: np.ndarray, rms_tol: float) -> bool:
    """
    Heuristic for 'output is bit-identical-or-quantized-copy of input'.
    Ignores PCM_16 quantization noise (~1.5e-5 rms) by default.
    """
    if len(in_samples) != len(out_samples):
        return False
    rms_in = _rms(in_samples)
    rms_out = _rms(out_samples)
    if max(rms_in, rms_out) < 1e-6:
        return True  # both effectively silent
    rms_diff = abs(rms_out - rms_in) / max(rms_in, 1e-9)
    return rms_diff < rms_tol


def check_audio_changed(
    ds,
    input_audio_dir: str,
    sample_size: int,
    rms_unchanged_tol: float = 0.001,
) -> CheckResult:
    if not os.path.isdir(input_audio_dir):
        return CheckResult(
            name="V1 Audio changed",
            passed=False,
            summary=f"input audio dir does not exist: {input_audio_dir}",
        )

    n = len(ds)
    indices = list(range(n))
    random.Random(0).shuffle(indices)
    indices = indices[:sample_size]

    augmented_checked = 0
    augmented_unmodified = 0
    skip_checked = 0      # skip_prob + skip_long — intentional pass-through
    failed_checked = 0    # rows where the pipeline raised / produced NaN
    failed_modified = 0   # failed-fallback rows whose audio differs from source
    missing_source = 0
    decode_errors = 0
    examples: list[str] = []

    for idx in indices:
        row = ds[idx]
        status = row.get("aug_status")
        # invalid_input has no useful source comparison; unknown statuses
        # don't come from our pipeline. Skip both.
        if status not in ("augmented", "skip_prob", "skip_long", "failed"):
            continue

        audio = row.get("audio")
        if not isinstance(audio, dict):
            continue
        path = audio.get("path") or ""
        base = os.path.basename(path)
        if not base:
            continue

        src_path = os.path.join(input_audio_dir, _strip_prefix(base))
        if not os.path.exists(src_path):
            missing_source += 1
            continue

        try:
            out_samples, _ = _decode_audio_entry(audio)
            import soundfile as sf
            in_samples, _ = sf.read(src_path)
        except Exception:
            decode_errors += 1
            continue

        out_samples = np.asarray(out_samples).flatten().astype(np.float32)
        in_samples = np.asarray(in_samples).flatten().astype(np.float32)

        unmodified = _looks_unmodified(in_samples, out_samples, rms_unchanged_tol)

        if status == "augmented":
            # Successful augmentation must materially change the waveform.
            augmented_checked += 1
            if unmodified:
                augmented_unmodified += 1
                if len(examples) < 3:
                    examples.append(base)
        elif status in ("skip_prob", "skip_long"):
            # Intentional pass-through — count, but don't flag modifications:
            # this is the designed behavior, not an error condition.
            skip_checked += 1
        else:  # status == "failed"
            # Pipeline raised or produced NaN/Inf; the fallback path must be
            # a clean re-encode of source. A modified fallback is suspicious
            # (e.g. partial augmentation leaked through the catch).
            failed_checked += 1
            if not _looks_unmodified(in_samples, out_samples, rms_tol=0.05):
                failed_modified += 1

    passed = augmented_unmodified == 0 and augmented_checked > 0

    # Warning-level signal: failed-fallback rows whose audio differs from source.
    # This does NOT affect PASS/FAIL — the contract is unchanged — but a
    # silent fallback that mutates audio can mean partial-augmentation leaked
    # through an exception catch, which the operator should know about.
    failed_modified_ratio = (
        failed_modified / failed_checked if failed_checked > 0 else 0.0
    )

    # Critical-level signal: when more than half of the failed-fallback rows
    # are mutated, the catch-and-fallback path is itself corrupting data.
    # The failed_checked >= 5 guard prevents single/few-sample false positives
    # (e.g. 1/1 = 100% on a tiny sample is statistically meaningless).
    # This does NOT alter passed; it embeds a [CRITICAL] token in the summary
    # which main() escalates into a fatal exit code.
    critical_failure = (
        failed_checked >= 5 and failed_modified_ratio > 0.5
    )

    summary = (
        f"{augmented_unmodified}/{augmented_checked} 'augmented' rows look "
        f"identical to source"
        + (" (no 'augmented' rows found in sample)" if augmented_checked == 0 else "")
    )
    # WARNING and CRITICAL are mutually exclusive — never emit both for the
    # same V1 result, so downstream parsers see exactly one severity tag.
    if critical_failure:
        summary += (
            f"; CRITICAL: high failed fallback corruption rate: "
            f"{failed_modified}/{failed_checked} ({failed_modified_ratio:.1%}) [CRITICAL]"
        )
    elif failed_modified_ratio > 0:
        summary += (
            f"; WARNING: {failed_modified}/{failed_checked} failed rows "
            f"produced modified audio ({failed_modified_ratio:.1%})"
        )

    return CheckResult(
        name="V1 Audio changed",
        passed=passed,
        summary=summary,
        metrics={
            "sampled_indices": len(indices),
            "augmented_checked": augmented_checked,
            "augmented_unmodified": augmented_unmodified,
            "skip_checked": skip_checked,
            "failed_checked": failed_checked,
            "failed_modified": failed_modified,
            "failed_modified_ratio": (
                round(failed_modified_ratio, 4) if failed_checked > 0 else 0.0
            ),
            "source_missing": missing_source,
            "decode_errors": decode_errors,
            "examples": examples,
        },
    )


# ─────────────────────────────────────────
# Driver
# ─────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the augmented dataset.")
    p.add_argument("--output-ds",
                   default=os.environ.get("OUTPUT_DS", "birlashtirilgan_dataset_augmented"))
    p.add_argument("--musan-path",
                   default=os.environ.get("MUSAN_PATH", "./musan"))
    p.add_argument("--input-audio-dir",
                   default=os.environ.get("INPUT_AUDIO_DIR", ""),
                   help="Directory containing the pre-augmentation audio files (Section 4 output). "
                        "Required for V1; if empty, V1 is skipped.")
    p.add_argument("--aug-prob", type=float,
                   default=float(os.environ.get("AUG_PROB", "1.0")),
                   help="Expected fraction of effective rows (total minus invalid_input) "
                        "with aug_status='augmented'.")
    p.add_argument("--prob-tolerance", type=float, default=0.10,
                   help="Allowed deviation between observed augmented ratio and --aug-prob.")
    p.add_argument("--max-failed-ratio", type=float, default=0.02,
                   help="Maximum tolerated fraction of rows with aug_status='failed' "
                        "(default 0.02 = 2%%). Computed over total_rows.")
    p.add_argument("--max-invalid-ratio", type=float, default=0.01,
                   help="Maximum tolerated fraction of rows with aug_status='invalid_input' "
                        "(default 0.01 = 1%%). Computed over total_rows.")
    p.add_argument("--min-musan-files", type=int, default=50,
                   help="Minimum .wav files required in each MUSAN subdir for V7.")
    p.add_argument("--sample-size", type=int, default=200,
                   help="Number of rows to sample for V1.")
    p.add_argument("--decode-limit", type=int, default=0,
                   help="Number of rows to scan in V5 (0 = all).")
    p.add_argument("--auto-filter-failed", action="store_true",
                   help="Before validation, drop rows with aug_status='failed' from the "
                        "in-memory dataset. Does NOT modify on-disk data.")
    p.add_argument("--strict-augmented-only", action="store_true",
                   help="Before validation, keep ONLY rows with aug_status='augmented'. "
                        "Stricter than --auto-filter-failed. Does NOT modify on-disk data.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("Augmentation Validation Report")
    print("=" * 60)
    print(f"  OUTPUT_DS:        {args.output_ds}")
    print(f"  MUSAN_PATH:       {args.musan_path}")
    print(f"  INPUT_AUDIO_DIR:  {args.input_audio_dir or '(not provided — V1 will be skipped)'}")
    print(f"  AUG_PROB:         {args.aug_prob}")
    print(f"  MIN_MUSAN_FILES:  {args.min_musan_files}")
    print("=" * 60)

    results: list[CheckResult] = []

    # V7 first — preflight, dataset-independent
    results.append(check_musan_preflight(args.musan_path, args.min_musan_files))

    # Load dataset for V3/V4/V5/V1
    try:
        from datasets import load_from_disk
        ds = load_from_disk(args.output_ds)
        print(f"\nLoaded {len(ds):,} rows from {args.output_ds}\n")
    except Exception as e:
        print(f"\n[FAIL] Could not load dataset {args.output_ds!r}: {e}\n")
        for r in results:
            print(r.report() + "\n")

        n_pass = sum(1 for r in results if r.passed)
        n_fail = sum(1 for r in results if not r.passed)
        payload = {
            "status": "fail",
            "checks_passed": n_pass,
            "checks_failed": n_fail,
            "total_checks": len(results),
            "reason": "dataset_load_error",
        }
        print(f"RESULT_JSON: {json.dumps(payload)}")
        return 2

    # Optional in-memory filtering. The on-disk dataset is never modified.
    # Strict mode is more restrictive than auto-filter, so it takes precedence
    # when both are set.
    if args.strict_augmented_only and args.auto_filter_failed:
        print("[WARN] Both filters set; strict_augmented_only takes precedence")
    if args.strict_augmented_only:
        ds = ds.filter(lambda x: x.get("aug_status") == "augmented")
        print("[INFO] Strict mode: using only fully augmented samples")
    elif args.auto_filter_failed:
        ds = ds.filter(lambda x: x.get("aug_status") != "failed")
        print("[INFO] Auto-filter enabled: removed rows with aug_status='failed'")

    if args.strict_augmented_only or args.auto_filter_failed:
        print("[WARN] Validation is running on a filtered dataset; metrics may be optimistic")

    if len(ds) == 0:
        print("[ERROR] Dataset is empty after filtering; aborting validation")

        payload = {
            "status": "fail",
            "checks_passed": 0,
            "checks_failed": 0,
            "total_checks": 0,
            "reason": "empty_after_filter"
        }

        print(f"RESULT_JSON: {json.dumps(payload)}")
        return 2

    results.append(check_schema(ds))
    results.append(check_status_distribution(
        ds,
        args.aug_prob,
        args.prob_tolerance,
        failed_ratio_threshold=args.max_failed_ratio,
        invalid_input_ratio_threshold=args.max_invalid_ratio,
    ))
    results.append(check_decode_all(ds, args.decode_limit or None))

    if args.input_audio_dir:
        results.append(check_audio_changed(ds, args.input_audio_dir, args.sample_size))
    else:
        print("[SKIP] V1 — no --input-audio-dir provided\n")

    for r in results:
        print(r.report() + "\n")

    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)
    overall_pass = all(r.passed for r in results)
    # Critical detection is metrics-based, not string-based: read the same
    # numeric signals V1 used to derive its [CRITICAL] tag, so the decision
    # is deterministic and immune to summary-text drift.
    critical_detected = any(
        r.name == "V1 Audio changed" and
        r.metrics.get("failed_checked", 0) >= 5 and
        r.metrics.get("failed_modified_ratio", 0) > 0.5
        for r in results
    )

    print("=" * 60)
    print(f"Summary: {n_pass} passed, {n_fail} failed (of {len(results)} checks run)")
    print("=" * 60)

    # Machine-readable line for CI/CD parsing. Always emitted, on a single
    # line, with a fixed "RESULT_JSON:" prefix that downstream tooling can
    # grep for. Human-readable logs above are unaffected.
    result_payload = {
        "status": "critical" if critical_detected else ("fail" if not overall_pass else "pass"),
        "checks_passed": n_pass,
        "checks_failed": n_fail,
        "total_checks": len(results),
    }
    print("RESULT_JSON:", json.dumps(result_payload))

    if critical_detected:
        print("[FATAL] Critical data corruption detected — dataset is NOT safe for training")
        return 3

    if not overall_pass:
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
