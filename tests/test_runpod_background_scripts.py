from __future__ import annotations

import subprocess
from pathlib import Path


def test_runpod_common_autoloads_sibling_runpod_env(tmp_path: Path) -> None:
    runpod_dir = tmp_path / "scripts" / "runpod"
    runpod_dir.mkdir(parents=True)
    common_src = (Path(__file__).resolve().parents[1] / "scripts" / "runpod" / "common.sh").read_text(encoding="utf-8")
    common_path = runpod_dir / "common.sh"
    common_path.write_text(common_src, encoding="utf-8")
    (runpod_dir / "runpod.env").write_text("export USE_HF=1\nexport FINO_CONVERT_LOG=/tmp/custom_convert.log\n", encoding="utf-8")
    cmd = [
        "bash",
        "-lc",
        (
            f'unset USE_HF FINO_CONVERT_LOG EDSVFH_RUNPOD_ENV_LOADED PROJECT_DIR PYTHONPATH; '
            f'source "{common_path}"; '
            f'printf "%s\n%s\n%s\n" "$USE_HF" "$FINO_CONVERT_LOG" "$PROJECT_DIR"'
        ),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == "1"
    assert lines[1] == "/tmp/custom_convert.log"
    assert lines[2] == str(tmp_path)


def test_runpod_background_launchers_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runpod_dir = repo_root / "scripts" / "runpod"
    expected = [
        "06b_launch_generate_fino_manifest_bg.sh",
        "06b_launch_generate_droid_failure_manifest_bg.sh",
        "07a_launch_fit_droid_success_baseline_bg.sh",
        "07b_launch_label_fino_pseudo_onset_bg.sh",
        "07b_launch_label_droid_failure_pseudo_onset_bg.sh",
        "07_launch_convert_fino_bg.sh",
        "07_launch_convert_droid_failure_bg.sh",
        "07c_launch_fino_pseudo_onset_pipeline_bg.sh",
        "07c_launch_droid_failure_pseudo_onset_pipeline_bg.sh",
        "08_launch_finetune_fino_bg.sh",
        "08_launch_finetune_droid_failure_bg.sh",
        "20b_launch_clean_rerun_fino_pseudo_onset_bg.sh",
        "21b_launch_validate_clean_rerun_canary_bg.sh",
        "22b_launch_test_larger_horizons_bg.sh",
        "23b_launch_eval_replay_protocol_bg.sh",
        "24b_launch_eval_ablation_suite_bg.sh",
        "00b_launch_bootstrap_bg.sh",
        "13b_launch_full_droid_then_droid_failure_bg.sh",
        "25b_launch_full_paper_path_droid_failure_bg.sh",
        "30b_launch_smoke_test_full_paper_path_droid_failure_bg.sh",
    ]
    for name in expected:
        text = (runpod_dir / name).read_text(encoding="utf-8")
        assert "launch_bg_named" in text
        assert "PID=" in text


def test_runpod_canary_validation_scripts_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runpod_dir = repo_root / "scripts" / "runpod"
    script = (runpod_dir / "21_validate_clean_rerun_canary.sh").read_text(encoding="utf-8")
    assert "test_convert_fino_manifest_rebuilds_when_manifest_changes" in script
    assert "test_pseudo_onset_keeps_original_when_low_confidence" in script
    assert "test_rebuild_fino_pseudo_onset_runs_end_to_end" in script
    assert "CANARY_PSEUDO_ONSET_FIT_MAX_EPISODES" in script
    launcher = (runpod_dir / "21b_launch_validate_clean_rerun_canary_bg.sh").read_text(encoding="utf-8")
    assert "launch_bg_named" in launcher
    assert "SUMMARY_JSON=" in launcher


def test_fino_finetune_script_supports_update_scaler() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts" / "runpod" / "08_finetune_fino.sh").read_text(encoding="utf-8")
    assert "FINO_UPDATE_SCALER" in script
    assert "--update-scaler" in script
    assert "--horizons" in script


def test_larger_horizons_script_is_packaged() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runpod_dir = repo_root / "scripts" / "runpod"
    script = (runpod_dir / "22_test_larger_horizons.sh").read_text(encoding="utf-8")
    assert "LARGER_HORIZONS" in script
    assert "--horizons" in script
    assert "LARGER_HORIZONS_OUTPUT_BUNDLE" in script
    assert "LARGER_HORIZONS_BASE_BUNDLE" in script
    assert "--freeze-existing-horizons" in script
    assert "LARGER_HORIZONS_FREEZE_EXISTING" in script


def test_v30_eval_scripts_are_packaged() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runpod_dir = repo_root / "scripts" / "runpod"
    replay = (runpod_dir / "23_eval_replay_protocol.sh").read_text(encoding="utf-8")
    ablation = (runpod_dir / "24_eval_ablation_suite.sh").read_text(encoding="utf-8")
    env = (runpod_dir / "runpod.env").read_text(encoding="utf-8")
    assert "edsvfh.eval_protocols replay" in replay
    assert "REPLAY_FIXED_RATES" in replay
    assert "edさvfh.eval_protocols" not in replay
    assert "edsvfh.eval_protocols ablation" in ablation
    assert "ABLATION_BUNDLES" in ablation
    assert "REPLAY_OUTPUT_JSON" in env
    assert "ABLATION_OUTPUT_JSON" in env


def test_droid_failure_runpod_scripts_are_packaged() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runpod_dir = repo_root / "scripts" / "runpod"
    manifest = (runpod_dir / "06_generate_droid_failure_manifest.sh").read_text(encoding="utf-8")
    label = (runpod_dir / "07b_label_droid_failure_pseudo_onset.sh").read_text(encoding="utf-8")
    convert = (runpod_dir / "07_convert_droid_failure.sh").read_text(encoding="utf-8")
    finetune = (runpod_dir / "08_finetune_droid_failure.sh").read_text(encoding="utf-8")
    env = (runpod_dir / "runpod.env").read_text(encoding="utf-8")
    assert "generate-droid-failure-manifest" in manifest
    assert "label-droid-failure-pseudo-onset" in label
    assert '--encoder "${ENCODER}"' in label
    assert "convert-droid-failure-manifest" in convert
    assert "fine-tune-droid-failure" in finetune
    assert "DROID_FAILURE_RAW_ROOT" in env
    assert "DROID_FAILURE_UPDATE_SCALER" in finetune
    assert "--update-scaler" in finetune



def test_v33_runpod_env_is_dynamic() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_text = (repo_root / "scripts" / "runpod" / "runpod.env").read_text(encoding="utf-8")
    assert "BASH_SOURCE[0]" in env_text
    assert "DROID_FAILURE_SCAN_MAX_EPISODES" in env_text
    assert "EDSVFH_FORCE_RERUN" in env_text


def test_v33_droid_failure_eval_scripts_target_droid_failure_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runpod_dir = repo_root / "scripts" / "runpod"
    larger = (runpod_dir / "22_test_larger_horizons.sh").read_text(encoding="utf-8")
    replay = (runpod_dir / "23_eval_replay_protocol.sh").read_text(encoding="utf-8")
    ablation = (runpod_dir / "24_eval_ablation_suite.sh").read_text(encoding="utf-8")
    assert "fine-tune-droid-failure" in larger
    assert "DROID_FAILURE_CONVERTED_ROOT" in larger
    assert "DROID_FAILURE_CONVERTED_ROOT" in replay
    assert "DROID_FAILURE_CONVERTED_ROOT" in ablation
    assert "fine-tune-fino" not in larger


def test_v33_smoke_script_is_isolated() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts" / "runpod" / "30_smoke_test_full_paper_path_droid_failure.sh").read_text(encoding="utf-8")
    assert "smoke_droid_curated" in script
    assert 'export DROID_FAILURE_SCAN_MAX_EPISODES="256"' in script
    assert "paper_pack_smoke" in script


def test_v34_label_scripts_and_status_include_runtime_monitoring() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runpod_dir = repo_root / "scripts" / "runpod"
    label_fino = (runpod_dir / "07b_label_fino_pseudo_onset.sh").read_text(encoding="utf-8")
    verify = (runpod_dir / "12_verify_project.sh").read_text(encoding="utf-8")
    status = (runpod_dir / "17_status_and_logs.sh").read_text(encoding="utf-8")
    assert '--encoder "${ENCODER}"' in label_fino
    assert '07c_run_droid_failure_pseudo_onset_pipeline.sh' in verify
    assert 'label-droid-failure-pseudo-onset' in verify
    assert 'full_paper_path_droid_failure.log' in status
    assert 'smoke_test_full_paper_path_droid_failure.log' in status
