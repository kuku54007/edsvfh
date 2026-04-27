from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from .api import create_app
from .config import AppConfig, DEFAULT_BUNDLE_PATH, DEFAULT_FIXTURE_PATH
from .models import VerifierBundle
from .pipeline import EventDrivenVerifierPipeline
from .droid_convert import DroidPreparedTFDSSource, create_mock_droid_shards, convert_droid_source_to_shards
from .droid_failure import generate_droid_failure_manifest_from_raw, generate_droid_failure_manifest_from_rlds, rebuild_droid_failure_with_pseudo_onset
from .fino_convert import convert_failure_manifest_to_shards, create_mock_failure_manifest_dataset
from .fino_finetune import fine_tune_bundle_on_failure_shards
from .fiper_pseudo_onset import fit_droid_success_baseline, relabel_fino_manifest_with_pseudo_onsets, rebuild_fino_with_pseudo_onset
from .manifest_tools import generate_fino_manifest_from_episode_dirs
from .public_data import create_tiny_robomimic_fixture, dataset_catalog, load_robomimic_hdf5
from .sharded_train import train_bundle_from_shards
from .train_public import train_from_robomimic, train_on_fixture



def _parse_horizons_arg(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or str(raw).strip() == '':
        return None
    values: list[int] = []
    for token in str(raw).replace(';', ',').replace(' ', ',').split(','):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f'horizon values must be positive integers, got {value}')
        values.append(value)
    return tuple(sorted(dict.fromkeys(values))) if values else None

def _metric_notes(metrics: dict) -> list[str]:
    notes: list[str] = []
    nan_horizons = [k for k, v in metrics.items() if k.startswith("horizon_") and str(v) == "nan"]
    if nan_horizons:
        notes.append("NaN horizon AUC usually means the evaluation split contains only a single class for that horizon; use FINO or another failure-rich evaluation set for failure-horizon validation.")
    return notes


def cmd_catalog(args: argparse.Namespace) -> int:
    print(json.dumps(dataset_catalog(), indent=2, ensure_ascii=False))
    return 0




def cmd_convert_mock_droid(args: argparse.Namespace) -> int:
    manifest = create_mock_droid_shards(
        args.output_dir,
        num_episodes=args.num_episodes,
        steps_per_episode=args.steps_per_episode,
        episodes_per_shard=args.episodes_per_shard,
        image_size=args.image_size,
        include_failures=not args.success_only,
        seed=args.seed,
    )
    print(json.dumps(manifest.as_dict(), indent=2))
    return 0


def cmd_convert_droid(args: argparse.Namespace) -> int:
    source = DroidPreparedTFDSSource(
        source=args.source,
        split=args.split,
        dataset_name=args.dataset_name,
        version=args.version,
        max_episodes=args.max_episodes,
    )
    manifest = convert_droid_source_to_shards(
        source,
        args.output_dir,
        source_label=str(args.source),
        source_mode='prepared_tfds',
        episodes_per_shard=args.episodes_per_shard,
        image_size=args.image_size,
        step_stride=args.step_stride,
        action_space=args.action_space,
        compression=None if args.no_compression else 'gzip',
        show_progress=not args.no_progress,
        resume=not args.no_resume,
        checkpoint_every=args.checkpoint_every,
        precompute_encoder=args.precompute_encoder,
        precompute_device=args.precompute_device,
        outcome_filter=args.outcome_filter,
    )
    print(json.dumps(manifest.as_dict(), indent=2))
    return 0






def cmd_fit_droid_success_baseline(args: argparse.Namespace) -> int:
    config = AppConfig()
    config.encoder.name = args.encoder
    baseline = fit_droid_success_baseline(
        args.shard_dir,
        output_path=args.output,
        config=config,
        feature_source=args.feature_source,
        window=args.window,
        phase_bins=args.phase_bins,
        quantile=args.quantile,
        min_phase_count=args.min_phase_count,
        max_episodes=args.max_episodes,
        show_progress=not args.no_progress,
    )
    print(json.dumps({'baseline': str(args.output), 'description': baseline.describe()}, indent=2, ensure_ascii=False))
    return 0


def cmd_label_fino_pseudo_onset(args: argparse.Namespace) -> int:
    config = AppConfig()
    if getattr(args, 'encoder', None):
        config.encoder.name = args.encoder
    result = relabel_fino_manifest_with_pseudo_onsets(
        args.manifest,
        args.baseline,
        args.output,
        image_size=args.image_size,
        replace_failure_onset=not args.no_replace_failure_onset,
        config=config,
        show_progress=not args.no_progress,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
    print(json.dumps({
        'baseline': result.baseline_path,
        'manifest': result.output_path,
        'total_episodes': result.total_episodes,
        'failure_episodes': result.failure_episodes,
        'pseudo_labeled_failures': result.pseudo_labeled_failures,
        'success_episodes': result.success_episodes,
        'replaced_failure_onsets': result.replaced_failure_onsets,
        'preserved_original_failure_onsets': result.preserved_original_failure_onsets,
        'checkpoint': result.checkpoint_path,
    }, indent=2, ensure_ascii=False))
    return 0



def cmd_generate_droid_failure_manifest(args: argparse.Namespace) -> int:
    camera_preference = tuple(x.strip() for x in str(args.camera_preference).split(',') if x.strip())
    result = generate_droid_failure_manifest_from_raw(
        args.root_dir,
        args.output,
        frames_root=args.frames_root,
        image_size=args.image_size,
        frame_stride=args.frame_stride,
        max_episodes=args.max_episodes,
        max_frames_per_episode=args.max_frames_per_episode,
        camera_preference=camera_preference,
        overwrite_frames=args.overwrite_frames,
        show_progress=not args.no_progress,
    )
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    return 0



def cmd_generate_droid_rlds_failure_manifest(args: argparse.Namespace) -> int:
    camera_preference = tuple(x.strip() for x in str(args.camera_preference).split(',') if x.strip())
    result = generate_droid_failure_manifest_from_rlds(
        args.source,
        args.output,
        split=args.split,
        dataset_name=args.dataset_name,
        version=args.version,
        frames_root=args.frames_root,
        image_size=args.image_size,
        frame_stride=args.frame_stride,
        max_episodes=args.max_episodes,
        scan_max_episodes=args.scan_max_episodes,
        max_frames_per_episode=args.max_frames_per_episode,
        camera_preference=camera_preference,
        overwrite_frames=args.overwrite_frames,
        show_progress=not args.no_progress,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    return 0

def cmd_label_droid_failure_pseudo_onset(args: argparse.Namespace) -> int:
    config = AppConfig()
    if getattr(args, 'encoder', None):
        config.encoder.name = args.encoder
    result = relabel_fino_manifest_with_pseudo_onsets(
        args.manifest,
        args.baseline,
        args.output,
        image_size=args.image_size,
        replace_failure_onset=not args.no_replace_failure_onset,
        config=config,
        show_progress=not args.no_progress,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
    print(json.dumps({
        'source_dataset': 'DROID_not_successful',
        'baseline': result.baseline_path,
        'manifest': result.output_path,
        'total_episodes': result.total_episodes,
        'failure_episodes': result.failure_episodes,
        'pseudo_labeled_failures': result.pseudo_labeled_failures,
        'success_episodes': result.success_episodes,
        'replaced_failure_onsets': result.replaced_failure_onsets,
        'preserved_original_failure_onsets': result.preserved_original_failure_onsets,
        'checkpoint': result.checkpoint_path,
    }, indent=2, ensure_ascii=False))
    return 0


def cmd_convert_droid_failure_manifest(args: argparse.Namespace) -> int:
    manifest = convert_failure_manifest_to_shards(
        args.manifest,
        args.output_dir,
        source_label=str(args.manifest),
        source_mode='droid_not_successful_manifest',
        episodes_per_shard=args.episodes_per_shard,
        image_size=args.image_size,
        compression=None if args.no_compression else 'gzip',
        show_progress=not args.no_progress,
        checkpoint_path=args.checkpoint,
        checkpoint_every_shards=args.checkpoint_every,
        resume=not args.no_resume,
        prefer_pseudo_onset=args.prefer_pseudo_onset,
    )
    print(json.dumps(manifest.as_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_fine_tune_droid_failure(args: argparse.Namespace) -> int:
    config = AppConfig()
    config.encoder.name = args.encoder
    horizons = _parse_horizons_arg(getattr(args, 'horizons', None))
    if horizons is not None:
        config.training.horizons = horizons
    result = fine_tune_bundle_on_failure_shards(
        args.base_bundle,
        args.shard_dir,
        output_path=args.output,
        config=config,
        epochs=args.epochs,
        update_scaler=args.update_scaler,
        show_progress=not args.no_progress,
        checkpoint_path=args.checkpoint,
        checkpoint_every_shards=args.checkpoint_every,
        resume=not args.no_resume,
        freeze_existing_horizons=getattr(args, 'freeze_existing_horizons', False),
    )
    print(json.dumps({
        'source_dataset': 'DROID_not_successful',
        'bundle': str(args.output),
        'metrics': result.metrics,
        'notes': _metric_notes(result.metrics),
        'strategies': result.strategies,
        'checkpoint': result.checkpoint_path,
        'train_shards': result.train_shards,
        'calib_shards': result.calib_shards,
        'eval_shards': result.eval_shards,
    }, indent=2, ensure_ascii=False))
    return 0


def cmd_rebuild_droid_failure_pseudo_onset(args: argparse.Namespace) -> int:
    config = AppConfig()
    config.encoder.name = args.encoder
    horizons = _parse_horizons_arg(getattr(args, 'horizons', None))
    if horizons is not None:
        config.training.horizons = horizons
    result = rebuild_droid_failure_with_pseudo_onset(
        args.droid_success_shard_dir,
        args.droid_failure_manifest,
        args.base_bundle,
        baseline_output_path=args.baseline_output,
        pseudo_manifest_output_path=args.pseudo_manifest_output,
        converted_output_dir=args.converted_output_dir,
        output_bundle_path=args.output_bundle,
        config=config,
        epochs=args.epochs,
        feature_source=args.feature_source,
        window=args.window,
        phase_bins=args.phase_bins,
        quantile=args.quantile,
        min_phase_count=args.min_phase_count,
        image_size=args.image_size,
        update_scaler=args.update_scaler,
        show_progress=not args.no_progress,
        fit_max_episodes=args.fit_max_episodes,
        replace_failure_onset=not args.no_replace_failure_onset,
        prefer_pseudo_onset=not args.no_prefer_pseudo_onset,
        episodes_per_shard=args.episodes_per_shard,
        pseudo_checkpoint_path=args.pseudo_checkpoint,
        pseudo_checkpoint_every=args.pseudo_checkpoint_every,
        pseudo_resume=not args.no_resume,
        droid_failure_checkpoint_path=args.checkpoint,
        droid_failure_checkpoint_every_shards=args.checkpoint_every,
        droid_failure_resume=not args.no_resume,
    )
    print(json.dumps({
        'source_dataset': 'DROID_not_successful',
        'baseline': result.baseline_path,
        'pseudo_manifest': result.pseudo_manifest_path,
        'converted_root': result.converted_root,
        'bundle': result.output_bundle,
        'metrics': result.metrics,
        'notes': _metric_notes(result.metrics),
        'strategies': result.strategies,
    }, indent=2, ensure_ascii=False))
    return 0

def cmd_rebuild_fino_pseudo_onset(args: argparse.Namespace) -> int:
    config = AppConfig()
    config.encoder.name = args.encoder
    horizons = _parse_horizons_arg(getattr(args, 'horizons', None))
    if horizons is not None:
        config.training.horizons = horizons
    result = rebuild_fino_with_pseudo_onset(
        args.droid_shard_dir,
        args.fino_manifest,
        args.base_bundle,
        baseline_output_path=args.baseline_output,
        pseudo_manifest_output_path=args.pseudo_manifest_output,
        converted_output_dir=args.converted_output_dir,
        output_bundle_path=args.output_bundle,
        config=config,
        epochs=args.epochs,
        feature_source=args.feature_source,
        window=args.window,
        phase_bins=args.phase_bins,
        quantile=args.quantile,
        min_phase_count=args.min_phase_count,
        image_size=args.image_size,
        update_scaler=args.update_scaler,
        show_progress=not args.no_progress,
        fit_max_episodes=args.fit_max_episodes,
        replace_failure_onset=not args.no_replace_failure_onset,
        prefer_pseudo_onset=not args.no_prefer_pseudo_onset,
        pseudo_checkpoint_path=args.pseudo_checkpoint,
        pseudo_checkpoint_every=args.pseudo_checkpoint_every,
        pseudo_resume=not args.no_resume,
        fino_checkpoint_path=args.checkpoint,
        fino_checkpoint_every_shards=args.checkpoint_every,
        fino_resume=not args.no_resume,
    )
    print(json.dumps({
        'baseline': result.baseline_path,
        'pseudo_manifest': result.pseudo_manifest_path,
        'converted_root': result.converted_root,
        'bundle': result.output_bundle,
        'metrics': result.metrics,
        'notes': _metric_notes(result.metrics),
        'strategies': result.strategies,
    }, indent=2, ensure_ascii=False))
    return 0

def cmd_convert_fino_manifest(args: argparse.Namespace) -> int:
    manifest = convert_failure_manifest_to_shards(
        args.manifest,
        args.output_dir,
        source_label=str(args.manifest),
        source_mode='fino_manifest',
        episodes_per_shard=args.episodes_per_shard,
        image_size=args.image_size,
        compression=None if args.no_compression else 'gzip',
        show_progress=not args.no_progress,
        checkpoint_path=args.checkpoint,
        checkpoint_every_shards=args.checkpoint_every,
        resume=not args.no_resume,
        prefer_pseudo_onset=args.prefer_pseudo_onset,
    )
    print(json.dumps(manifest.as_dict(), indent=2))
    return 0


def cmd_convert_mock_failure(args: argparse.Namespace) -> int:
    manifest = create_mock_failure_manifest_dataset(
        args.root_dir,
        args.output_dir,
        num_episodes=args.num_episodes,
        image_size=args.image_size,
        episodes_per_shard=args.episodes_per_shard,
        seed=args.seed,
    )
    print(json.dumps(manifest.as_dict(), indent=2))
    return 0


def cmd_generate_fino_manifest(args: argparse.Namespace) -> int:
    path = generate_fino_manifest_from_episode_dirs(
        args.root_dir,
        args.output,
        frame_glob=args.frame_glob,
        task=args.task,
        instruction=args.instruction,
        show_progress=not args.no_progress,
        checkpoint_path=args.checkpoint,
        checkpoint_every_shards=args.checkpoint_every,
        resume=not args.no_resume,
    )
    print(json.dumps({'manifest': str(path)}, indent=2, ensure_ascii=False))
    return 0


def cmd_fine_tune_fino(args: argparse.Namespace) -> int:
    config = AppConfig()
    config.encoder.name = args.encoder
    horizons = _parse_horizons_arg(getattr(args, 'horizons', None))
    if horizons is not None:
        config.training.horizons = horizons
    result = fine_tune_bundle_on_failure_shards(
        args.base_bundle,
        args.shard_dir,
        output_path=args.output,
        config=config,
        epochs=args.epochs,
        update_scaler=args.update_scaler,
        show_progress=not args.no_progress,
        checkpoint_path=args.checkpoint,
        checkpoint_every_shards=args.checkpoint_every,
        resume=not args.no_resume,
        freeze_existing_horizons=getattr(args, 'freeze_existing_horizons', False),
    )
    print(json.dumps({
        'bundle': str(args.output),
        'metrics': result.metrics,
        'notes': _metric_notes(result.metrics),
        'strategies': result.strategies,
        'checkpoint': result.checkpoint_path,
        'train_shards': result.train_shards,
        'calib_shards': result.calib_shards,
        'eval_shards': result.eval_shards,
    }, indent=2))
    return 0

def cmd_train_sharded(args: argparse.Namespace) -> int:
    config = AppConfig()
    config.encoder.name = args.encoder
    horizons = _parse_horizons_arg(getattr(args, 'horizons', None))
    if horizons is not None:
        config.training.horizons = horizons
    result = train_bundle_from_shards(
        args.shard_dir,
        output_path=args.output,
        config=config,
        epochs=args.epochs,
        delete_consumed_train_shards=args.delete_consumed_train_shards,
        show_progress=not args.no_progress,
        checkpoint_path=args.checkpoint,
        checkpoint_every_shards=args.checkpoint_every,
        resume=not args.no_resume,
    )
    print(json.dumps({'bundle': str(args.output), 'metrics': result.metrics, 'notes': _metric_notes(result.metrics), 'checkpoint': result.checkpoint_path, 'train_shards': result.train_shards, 'calib_shards': result.calib_shards, 'eval_shards': result.eval_shards}, indent=2))
    return 0

def cmd_make_fixture(args: argparse.Namespace) -> int:
    path = create_tiny_robomimic_fixture(args.output)
    print(json.dumps({'fixture': str(path)}, indent=2))
    return 0


def cmd_train_fixture(args: argparse.Namespace) -> int:
    config = AppConfig()
    config.encoder.name = args.encoder
    bundle, metrics, fixture = train_on_fixture(output_path=args.output, fixture_path=args.fixture, config=config)
    print(json.dumps({'bundle': str(args.output), 'fixture': str(fixture), 'metrics': metrics}, indent=2))
    return 0


def cmd_train_robomimic(args: argparse.Namespace) -> int:
    config = AppConfig()
    config.encoder.name = args.encoder
    bundle, metrics, _ = train_from_robomimic(args.dataset, output_path=args.output, config=config)
    print(json.dumps({'bundle': str(args.output), 'metrics': metrics}, indent=2))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    bundle = VerifierBundle.load(args.bundle)
    pipeline = EventDrivenVerifierPipeline(bundle=bundle, config=AppConfig())
    episodes = load_robomimic_hdf5(args.dataset)
    if args.episode_index >= len(episodes):
        raise IndexError(f'episode_index {args.episode_index} out of range; available episodes={len(episodes)}')
    episode = episodes[args.episode_index]
    summary = pipeline.run_episode(episode, stop_on_termination=not args.no_stop)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    app = create_app(bundle_path=args.bundle, config=AppConfig())
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='EDSV-FH public-dataset reference implementation')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('catalog', help='print the recommended public dataset catalog')
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser('make-fixture', help='create a tiny robomimic-compatible fixture')
    p.add_argument('--output', type=Path, default=DEFAULT_FIXTURE_PATH)
    p.set_defaults(func=cmd_make_fixture)

    p = sub.add_parser('train-fixture', help='train on the validated tiny robomimic fixture')
    p.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE_PATH)
    p.add_argument('--output', type=Path, default=DEFAULT_BUNDLE_PATH)
    p.add_argument('--encoder', default='fallback')
    p.set_defaults(func=cmd_train_fixture)

    p = sub.add_parser('train-robomimic', help='train on a robomimic-format HDF5 dataset')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, default=DEFAULT_BUNDLE_PATH)
    p.add_argument('--encoder', default='fallback')
    p.set_defaults(func=cmd_train_robomimic)

    p = sub.add_parser('convert-mock-droid', help='create validated mock DROID-style shards for smoke tests')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--num-episodes', type=int, default=18)
    p.add_argument('--steps-per-episode', type=int, default=20)
    p.add_argument('--episodes-per-shard', type=int, default=4)
    p.add_argument('--image-size', type=int, default=96)
    p.add_argument('--seed', type=int, default=5)
    p.add_argument('--success-only', action='store_true', help='disable injected mock failure episodes')
    p.set_defaults(func=cmd_convert_mock_droid)

    p = sub.add_parser('convert-droid', help='convert prepared DROID RLDS data into robomimic-compatible shards')
    p.add_argument('--source', required=True, help='prepared TFDS builder directory or dataset root')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--split', default='train')
    p.add_argument('--dataset-name', default=None, help='optional TFDS builder name, e.g. droid')
    p.add_argument('--version', default=None, help='optional TFDS builder version')
    p.add_argument('--max-episodes', type=int, default=None)
    p.add_argument('--episodes-per-shard', type=int, default=64)
    p.add_argument('--image-size', type=int, default=96)
    p.add_argument('--step-stride', type=int, default=1)
    p.add_argument('--action-space', choices=['raw_action', 'joint_position', 'joint_velocity'], default='raw_action')
    p.add_argument('--outcome-filter', choices=['all', 'success', 'failure'], default='all', help='keep only episodes with the selected inferred outcome')
    p.add_argument('--precompute-encoder', default='fallback', help='precompute frozen visual/context features during conversion; use siglip2_dinov2 for GPU-assisted conversion or none to disable')
    p.add_argument('--precompute-device', default=None, help='device override for precompute encoder, e.g. cuda')
    p.add_argument('--checkpoint-every', type=int, default=32)
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--no-compression', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    p.set_defaults(func=cmd_convert_droid)

    p = sub.add_parser('train-sharded', help='incrementally train on robomimic-compatible shard directories')
    p.add_argument('--shard-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, default=DEFAULT_BUNDLE_PATH)
    p.add_argument('--encoder', default='fallback')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--horizons', default=None, help='comma-separated failure horizons, e.g. 1,3,5,10,15')
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--checkpoint-every', type=int, default=1)
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--delete-consumed-train-shards', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    p.set_defaults(func=cmd_train_sharded)


    p = sub.add_parser('fit-droid-success-baseline', help='fit a FIPER-style normal baseline from success-only DROID rollouts')
    p.add_argument('--shard-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--encoder', default='fallback')
    p.add_argument('--feature-source', choices=['visual', 'vector'], default='visual')
    p.add_argument('--window', type=int, default=3)
    p.add_argument('--phase-bins', type=int, default=10)
    p.add_argument('--quantile', type=float, default=0.97)
    p.add_argument('--min-phase-count', type=int, default=8)
    p.add_argument('--max-episodes', type=int, default=None)
    p.add_argument('--no-progress', action='store_true')
    p.set_defaults(func=cmd_fit_droid_success_baseline)


    p = sub.add_parser('generate-droid-failure-manifest', help='scan DROID raw not-successful/failure episodes and emit a failure manifest')
    p.add_argument('--root-dir', type=Path, required=True, help='DROID raw root; official raw releases can be filtered via metadata success flags or success/failure path names')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames-root', type=Path, default=None, help='where extracted RGB PNG frames and low-dimensional arrays are written')
    p.add_argument('--image-size', type=int, default=96)
    p.add_argument('--frame-stride', type=int, default=2)
    p.add_argument('--max-episodes', type=int, default=None)
    p.add_argument('--max-frames-per-episode', type=int, default=None)
    p.add_argument('--camera-preference', default='exterior_image_1_left,exterior_image_2_left,wrist_image_left,ext1,ext2,wrist')
    p.add_argument('--overwrite-frames', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    p.set_defaults(func=cmd_generate_droid_failure_manifest)


    p = sub.add_parser('generate-droid-rlds-failure-manifest', help='scan prepared DROID RLDS/TFDS data and emit a not-successful/failure manifest using metadata path tokens')
    p.add_argument('--source', required=True, help='prepared DROID TFDS builder directory, e.g. /workspace/data/raw/droid/1.0.1')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames-root', type=Path, default=None, help='where extracted RGB PNG frames and low-dimensional arrays are written')
    p.add_argument('--split', default='train')
    p.add_argument('--dataset-name', default=None, help='optional TFDS builder name')
    p.add_argument('--version', default=None, help='optional TFDS builder version')
    p.add_argument('--image-size', type=int, default=96)
    p.add_argument('--frame-stride', type=int, default=2)
    p.add_argument('--max-episodes', type=int, default=None, help='maximum number of failure episodes to keep')
    p.add_argument('--scan-max-episodes', type=int, default=None, help='maximum number of RLDS episodes to scan before stopping')
    p.add_argument('--max-frames-per-episode', type=int, default=None)
    p.add_argument('--camera-preference', default='exterior_image_1_left,exterior_image_2_left,wrist_image_left,ext1,ext2,wrist')
    p.add_argument('--overwrite-frames', action='store_true')
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--checkpoint-every', type=int, default=256)
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    p.set_defaults(func=cmd_generate_droid_rlds_failure_manifest)

    p = sub.add_parser('label-droid-failure-pseudo-onset', help='infer pseudo-onsets for DROID not-successful episodes using a DROID success baseline')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image-size', type=int, default=96)
    p.add_argument('--encoder', default=None, help='optional encoder override used to relabel manifest episodes')
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--checkpoint-every', type=int, default=32)
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--no-replace-failure-onset', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    p.set_defaults(func=cmd_label_droid_failure_pseudo_onset)

    p = sub.add_parser('convert-droid-failure-manifest', help='convert a DROID not-successful manifest into robomimic-compatible failure shards')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--episodes-per-shard', type=int, default=32)
    p.add_argument('--image-size', type=int, default=96)
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--checkpoint-every', type=int, default=16)
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--prefer-pseudo-onset', action='store_true')
    p.add_argument('--no-compression', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    p.set_defaults(func=cmd_convert_droid_failure_manifest)

    p = sub.add_parser('fine-tune-droid-failure', help='fine-tune failure-horizon heads on DROID not-successful shards')
    p.add_argument('--base-bundle', type=Path, required=True)
    p.add_argument('--shard-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, default=DEFAULT_BUNDLE_PATH)
    p.add_argument('--encoder', default='fallback')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--horizons', default=None, help='comma-separated failure horizons, e.g. 1,3,5,10,15')
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--checkpoint-every', type=int, default=1)
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--update-scaler', action='store_true')
    p.add_argument('--freeze-existing-horizons', action='store_true', help='keep horizon heads that already exist in the base bundle; train/calibrate only newly requested horizons')
    p.add_argument('--no-progress', action='store_true')
    p.set_defaults(func=cmd_fine_tune_droid_failure)

    p = sub.add_parser('rebuild-droid-failure-pseudo-onset', help='fit DROID success baseline, relabel DROID not-successful episodes, reconvert, and fine-tune in one command')
    p.add_argument('--droid-success-shard-dir', type=Path, required=True)
    p.add_argument('--droid-failure-manifest', type=Path, required=True)
    p.add_argument('--base-bundle', type=Path, required=True)
    p.add_argument('--baseline-output', type=Path, required=True)
    p.add_argument('--pseudo-manifest-output', type=Path, required=True)
    p.add_argument('--converted-output-dir', type=Path, required=True)
    p.add_argument('--output-bundle', type=Path, required=True)
    p.add_argument('--encoder', default='fallback')
    p.add_argument('--feature-source', choices=['visual', 'vector'], default='visual')
    p.add_argument('--window', type=int, default=3)
    p.add_argument('--phase-bins', type=int, default=10)
    p.add_argument('--quantile', type=float, default=0.97)
    p.add_argument('--min-phase-count', type=int, default=8)
    p.add_argument('--fit-max-episodes', type=int, default=None)
    p.add_argument('--image-size', type=int, default=96)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--horizons', default=None, help='comma-separated failure horizons, e.g. 1,3,5,10,15')
    p.add_argument('--episodes-per-shard', type=int, default=32)
    p.add_argument('--pseudo-checkpoint', type=Path, default=None)
    p.add_argument('--pseudo-checkpoint-every', type=int, default=32)
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--checkpoint-every', type=int, default=1)
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--update-scaler', action='store_true')
    p.add_argument('--no-replace-failure-onset', action='store_true')
    p.add_argument('--no-prefer-pseudo-onset', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    p.set_defaults(func=cmd_rebuild_droid_failure_pseudo_onset)

    p = sub.add_parser('label-fino-pseudo-onset', help='infer pseudo-onsets for each FINO episode using a success-only normal baseline')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image-size', type=int, default=96)
    p.add_argument('--encoder', default=None, help='optional encoder override used to relabel manifest episodes')
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--checkpoint-every', type=int, default=32)
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--no-replace-failure-onset', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    p.set_defaults(func=cmd_label_fino_pseudo_onset)

    p = sub.add_parser('convert-fino-manifest', help='convert a FINO-style manifest into robomimic-compatible failure shards')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--episodes-per-shard', type=int, default=32)
    p.add_argument('--image-size', type=int, default=96)
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--checkpoint-every', type=int, default=16)
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--prefer-pseudo-onset', action='store_true')
    p.add_argument('--no-compression', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    p.set_defaults(func=cmd_convert_fino_manifest)

    p = sub.add_parser('convert-mock-failure', help='create validated mock failure shards via a FINO-style manifest route')
    p.add_argument('--root-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--num-episodes', type=int, default=18)
    p.add_argument('--episodes-per-shard', type=int, default=4)
    p.add_argument('--image-size', type=int, default=96)
    p.add_argument('--seed', type=int, default=13)
    p.set_defaults(func=cmd_convert_mock_failure)

    p = sub.add_parser('generate-fino-manifest', help='scan a FINO-style episode tree and emit a JSONL manifest')
    p.add_argument('--root-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frame-glob', default='*.png')
    p.add_argument('--task', default='fino_failure')
    p.add_argument('--instruction', default='Detect whether the manipulation is heading toward failure.')
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--checkpoint-every', type=int, default=32)
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    p.set_defaults(func=cmd_generate_fino_manifest)

    p = sub.add_parser('fine-tune-fino', help='fine-tune the failure-horizon heads on FINO-style failure shards')
    p.add_argument('--base-bundle', type=Path, required=True)
    p.add_argument('--shard-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, default=DEFAULT_BUNDLE_PATH)
    p.add_argument('--encoder', default='fallback')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--horizons', default=None, help='comma-separated failure horizons, e.g. 1,3,5,10,15')
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--checkpoint-every', type=int, default=1)
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--update-scaler', action='store_true')
    p.add_argument('--freeze-existing-horizons', action='store_true', help='keep horizon heads that already exist in the base bundle; train/calibrate only newly requested horizons')
    p.add_argument('--no-progress', action='store_true')
    p.set_defaults(func=cmd_fine_tune_fino)

    p = sub.add_parser('rebuild-fino-pseudo-onset', help='fit a DROID success baseline, relabel FINO with pseudo-onsets, reconvert, and fine-tune in one command')
    p.add_argument('--droid-shard-dir', type=Path, required=True)
    p.add_argument('--fino-manifest', type=Path, required=True)
    p.add_argument('--base-bundle', type=Path, required=True)
    p.add_argument('--baseline-output', type=Path, required=True)
    p.add_argument('--pseudo-manifest-output', type=Path, required=True)
    p.add_argument('--converted-output-dir', type=Path, required=True)
    p.add_argument('--output-bundle', type=Path, required=True)
    p.add_argument('--encoder', default='fallback')
    p.add_argument('--feature-source', choices=['visual', 'vector'], default='visual')
    p.add_argument('--window', type=int, default=3)
    p.add_argument('--phase-bins', type=int, default=10)
    p.add_argument('--quantile', type=float, default=0.97)
    p.add_argument('--min-phase-count', type=int, default=8)
    p.add_argument('--fit-max-episodes', type=int, default=None)
    p.add_argument('--image-size', type=int, default=96)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--horizons', default=None, help='comma-separated failure horizons, e.g. 1,3,5,10,15')
    p.add_argument('--pseudo-checkpoint', type=Path, default=None)
    p.add_argument('--pseudo-checkpoint-every', type=int, default=32)
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--checkpoint-every', type=int, default=1)
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--update-scaler', action='store_true')
    p.add_argument('--no-replace-failure-onset', action='store_true')
    p.add_argument('--no-prefer-pseudo-onset', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    p.set_defaults(func=cmd_rebuild_fino_pseudo_onset)

    p = sub.add_parser('demo', help='run one dataset episode through the event-driven pipeline')
    p.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE_PATH)
    p.add_argument('--dataset', type=Path, default=DEFAULT_FIXTURE_PATH)
    p.add_argument('--episode-index', type=int, default=0)
    p.add_argument('--no-stop', action='store_true', help='replay the full episode even after a terminal decision')
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser('serve', help='serve the FastAPI integration app')
    p.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE_PATH)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
