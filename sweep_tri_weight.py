"""Run resumable tri_weight experiments and aggregate their metrics."""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import toml


CHECKPOINT_IGNORED_TRAIN_KEYS = {
    'checkpoint_enabled',
    'resume_checkpoint',
    'checkpoint_dir',
}
PRIORITY_COLUMNS = [
    'rank',
    'is_best',
    'tri_weight',
    'status',
    'selection_metric',
    'selection_key',
    'selection_value',
    'accuracy',
    'roc_auc',
    'pr_auc',
    'f1_score',
    'mcc',
    'loss',
    'cls_loss',
    'tri_loss',
    'relation_ib_loss',
    'mode',
    'seed',
    'split_seed',
    'started_at',
    'finished_at',
    'duration_seconds',
    'exit_code',
    'run_dir',
    'checkpoint_dir',
    'metrics_file',
    'log_file',
    'config_signature',
    'error',
]


def now_iso():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def display_command(command):
    if os.name == 'nt':
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def non_negative_finite_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('weights must be finite and greater than or equal to 0')
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Sweep tri_weight values with isolated checkpoints and a result table.'
    )
    parser.add_argument('--config', default='config.toml', help='Training TOML file.')
    parser.add_argument(
        '--weights',
        nargs='+',
        type=non_negative_finite_float,
        help='Weights to test. Overrides tri_weight_sweep.weights in the TOML file.',
    )
    parser.add_argument(
        '--output-dir',
        help='Sweep output directory. Overrides tri_weight_sweep.output_dir.',
    )
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop after the first failed weight instead of continuing.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the planned commands without starting training or writing files.',
    )
    return parser.parse_args(argv)


def canonical_weight(value):
    decimal_value = Decimal(str(value))
    if decimal_value == 0:
        return '0'
    return format(decimal_value.normalize(), 'f')


def weight_slug(value):
    return canonical_weight(value).replace('-', 'm').replace('.', 'p')


def unique_weights(values):
    result = []
    seen = set()
    for value in values:
        value = non_negative_finite_float(value)
        key = canonical_weight(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def config_signature(cfg, tri_weight):
    effective_cfg = copy.deepcopy(cfg)
    effective_cfg['train']['tri_weight'] = float(tri_weight)
    train_cfg = {
        key: value
        for key, value in effective_cfg['train'].items()
        if key not in CHECKPOINT_IGNORED_TRAIN_KEYS
    }
    payload = {'model': effective_cfg['model'], 'train': train_cfg}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(tmp_path, path)


def load_records(path):
    path = Path(path)
    if not path.exists():
        return []
    try:
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f'Warning: could not read {path}: {exc}', file=sys.stderr)
        return []
    if isinstance(payload, dict):
        payload = payload.get('runs', [])
    return payload if isinstance(payload, list) else []


def load_completed_metrics(path, expected_signature, expected_weight):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get('status') != 'completed':
        return None
    if not isinstance(payload.get('metrics'), dict):
        return None
    try:
        selection_value = float(payload.get('selection_value'))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(selection_value):
        return None
    if payload.get('config_signature') != expected_signature:
        return None
    try:
        actual_weight = canonical_weight(payload['tri_weight'])
    except (KeyError, TypeError, ValueError):
        return None
    if actual_weight != canonical_weight(expected_weight):
        return None
    return payload


def record_from_metrics(payload, run_dir, metrics_file, log_file, exit_code=0):
    metrics = payload.get('metrics') or {}
    record = {
        'tri_weight': payload.get('tri_weight'),
        'status': 'completed',
        'selection_metric': payload.get('selection_metric'),
        'selection_key': payload.get('selection_key'),
        'selection_value': payload.get('selection_value'),
        'mode': payload.get('mode'),
        'seed': payload.get('seed'),
        'split_seed': payload.get('split_seed'),
        'started_at': payload.get('started_at'),
        'finished_at': payload.get('finished_at'),
        'duration_seconds': payload.get('duration_seconds'),
        'exit_code': exit_code,
        'run_dir': str(run_dir),
        'checkpoint_dir': payload.get('checkpoint_dir'),
        'metrics_file': str(metrics_file),
        'log_file': str(log_file),
        'config_signature': payload.get('config_signature'),
        'error': '',
    }
    for key, value in metrics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            record[key] = value
    return record


def refresh_ranking(records):
    for record in records:
        record.pop('rank', None)
        record['is_best'] = False

    completed = []
    for record in records:
        if record.get('status') != 'completed':
            continue
        try:
            value = float(record.get('selection_value'))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            completed.append((value, float(record['tri_weight']), record))

    completed.sort(key=lambda item: (-item[0], item[1]))
    for rank, (_, _, record) in enumerate(completed, start=1):
        record['rank'] = rank
        record['is_best'] = rank == 1
    return completed[0][2] if completed else None


def csv_columns(records):
    keys = {key for record in records for key in record}
    columns = [name for name in PRIORITY_COLUMNS if name in keys]
    columns.extend(sorted(keys.difference(columns)))
    return columns


def write_csv(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.tmp')
    columns = csv_columns(records)
    with tmp_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        if columns:
            writer.writeheader()
            writer.writerows(records)
    os.replace(tmp_path, path)


def persist_results(records, cfg_path, weights, results_json, results_csv, best_json):
    records.sort(key=lambda record: float(record['tri_weight']))
    best = refresh_ranking(records)
    payload = {
        'config_path': str(cfg_path),
        'requested_weights': weights,
        'updated_at': now_iso(),
        'runs': records,
    }
    atomic_json_dump(payload, results_json)
    write_csv(records, results_csv)
    if best is not None:
        atomic_json_dump(best, best_json)
    else:
        atomic_json_dump(
            {'status': 'no_completed_runs', 'updated_at': now_iso()},
            best_json,
        )
    return best


def upsert_record(records_by_weight, record):
    records_by_weight[canonical_weight(record['tri_weight'])] = record


def stream_subprocess(command, cwd, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    with log_path.open('a', encoding='utf-8') as log_file:
        header = f'\n[{now_iso()}] $ {display_command(command)}\n'
        log_file.write(header)
        log_file.flush()
        print(header, end='')
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log_file.write(line)
                log_file.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise


def resolve_path(path):
    expanded = os.path.expanduser(os.fspath(path))
    return Path(os.path.abspath(expanded))


def main(argv=None):
    args = parse_args(argv)
    project_dir = Path(__file__).resolve().parent
    cfg_path = resolve_path(args.config)
    cfg = toml.load(cfg_path)
    sweep_cfg = cfg.get('tri_weight_sweep', {})

    raw_weights = args.weights if args.weights is not None else sweep_cfg.get('weights')
    if not raw_weights:
        raise SystemExit(
            'No weights supplied. Use --weights or set tri_weight_sweep.weights in config.toml.'
        )
    weights = unique_weights(raw_weights)

    output_value = args.output_dir or sweep_cfg.get(
        'output_dir', 'sweep_results/tri_weight'
    )
    output_dir = resolve_path(output_value)
    results_json = output_dir / 'tri_weight_results.json'
    results_csv = output_dir / 'tri_weight_results.csv'
    best_json = output_dir / 'best_tri_weight.json'

    plans = []
    for weight in weights:
        run_dir = output_dir / 'runs' / f'tri_weight_{weight_slug(weight)}'
        plans.append({
            'weight': weight,
            'signature': config_signature(cfg, weight),
            'run_dir': run_dir,
            'checkpoint_dir': run_dir / 'checkpoints',
            'metrics_file': run_dir / 'metrics.json',
            'log_file': run_dir / 'train.log',
        })

    if args.dry_run:
        for plan in plans:
            command = [
                sys.executable,
                '-u',
                str(project_dir / 'main.py'),
                '--config',
                str(cfg_path),
                '--tri-weight',
                canonical_weight(plan['weight']),
                '--model-dir',
                str(plan['run_dir']),
                '--checkpoint-dir',
                str(plan['checkpoint_dir']),
                '--metrics-file',
                str(plan['metrics_file']),
            ]
            print(display_command(command))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_records = load_records(results_json)
    records_by_weight = {}
    for record in existing_records:
        try:
            weight = float(record['tri_weight'])
        except (KeyError, TypeError, ValueError):
            continue
        if record.get('config_signature') == config_signature(cfg, weight):
            upsert_record(records_by_weight, record)

    failures = 0
    for index, plan in enumerate(plans, start=1):
        weight = plan['weight']
        metrics_payload = load_completed_metrics(
            plan['metrics_file'],
            plan['signature'],
            weight,
        )
        if metrics_payload is not None:
            record = record_from_metrics(
                metrics_payload,
                plan['run_dir'],
                plan['metrics_file'],
                plan['log_file'],
            )
            upsert_record(records_by_weight, record)
            persist_results(
                list(records_by_weight.values()),
                cfg_path,
                weights,
                results_json,
                results_csv,
                best_json,
            )
            print(
                f'[{index}/{len(plans)}] tri_weight={canonical_weight(weight)} '
                'already completed; skipping.'
            )
            continue

        plan['run_dir'].mkdir(parents=True, exist_ok=True)
        started_at = now_iso()
        started_time = time.time()
        running_record = {
            'tri_weight': weight,
            'status': 'running',
            'selection_metric': cfg['train'].get('selection_metric', 'roc_auc'),
            'selection_key': None,
            'selection_value': None,
            'started_at': started_at,
            'finished_at': '',
            'duration_seconds': 0.0,
            'exit_code': '',
            'run_dir': str(plan['run_dir']),
            'checkpoint_dir': str(plan['checkpoint_dir']),
            'metrics_file': str(plan['metrics_file']),
            'log_file': str(plan['log_file']),
            'config_signature': plan['signature'],
            'error': '',
        }
        upsert_record(records_by_weight, running_record)
        persist_results(
            list(records_by_weight.values()),
            cfg_path,
            weights,
            results_json,
            results_csv,
            best_json,
        )

        command = [
            sys.executable,
            '-u',
            str(project_dir / 'main.py'),
            '--config',
            str(cfg_path),
            '--tri-weight',
            canonical_weight(weight),
            '--model-dir',
            str(plan['run_dir']),
            '--checkpoint-dir',
            str(plan['checkpoint_dir']),
            '--metrics-file',
            str(plan['metrics_file']),
        ]
        print(
            f'[{index}/{len(plans)}] Starting tri_weight={canonical_weight(weight)}'
        )

        try:
            exit_code = stream_subprocess(command, project_dir, plan['log_file'])
            metrics_payload = load_completed_metrics(
                plan['metrics_file'],
                plan['signature'],
                weight,
            )
            if exit_code == 0 and metrics_payload is not None:
                record = record_from_metrics(
                    metrics_payload,
                    plan['run_dir'],
                    plan['metrics_file'],
                    plan['log_file'],
                    exit_code=exit_code,
                )
                print(
                    f'Completed tri_weight={canonical_weight(weight)}: '
                    f"{record.get('selection_key')}={record.get('selection_value')}"
                )
            else:
                error = f'main.py exited with code {exit_code}'
                if exit_code == 0:
                    error += ' but did not produce a matching completed metrics file'
                record = dict(running_record)
                record.update({
                    'status': 'failed',
                    'finished_at': now_iso(),
                    'duration_seconds': time.time() - started_time,
                    'exit_code': exit_code,
                    'error': error,
                })
                failures += 1
                print(f'Failed tri_weight={canonical_weight(weight)}: {error}', file=sys.stderr)
        except KeyboardInterrupt:
            record = dict(running_record)
            record.update({
                'status': 'interrupted',
                'finished_at': now_iso(),
                'duration_seconds': time.time() - started_time,
                'exit_code': 130,
                'error': 'Interrupted by user; rerun the same command to resume.',
            })
            upsert_record(records_by_weight, record)
            persist_results(
                list(records_by_weight.values()),
                cfg_path,
                weights,
                results_json,
                results_csv,
                best_json,
            )
            print('\nSweep interrupted. Progress was saved.', file=sys.stderr)
            return 130
        except Exception as exc:
            record = dict(running_record)
            record.update({
                'status': 'failed',
                'finished_at': now_iso(),
                'duration_seconds': time.time() - started_time,
                'exit_code': '',
                'error': f'{type(exc).__name__}: {exc}',
            })
            failures += 1
            print(
                f'Failed tri_weight={canonical_weight(weight)}: {record["error"]}',
                file=sys.stderr,
            )

        upsert_record(records_by_weight, record)
        best = persist_results(
            list(records_by_weight.values()),
            cfg_path,
            weights,
            results_json,
            results_csv,
            best_json,
        )
        if record['status'] != 'completed' and args.stop_on_error:
            break

    records = list(records_by_weight.values())
    best = persist_results(
        records,
        cfg_path,
        weights,
        results_json,
        results_csv,
        best_json,
    )
    print(f'Results CSV: {results_csv}')
    print(f'Results JSON: {results_json}')
    if best is not None:
        print(
            'Best tri_weight: '
            f"{canonical_weight(best['tri_weight'])} "
            f"({best.get('selection_key')}={best.get('selection_value')})"
        )
        print(f'Best result JSON: {best_json}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
