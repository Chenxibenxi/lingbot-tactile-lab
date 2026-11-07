"""Correct derived report captions without changing frozen experiment code/data."""
from pathlib import Path
import json, hashlib, shutil
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
DEST = Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m6-confirmation')


def correct():
    corrected = []
    for trial in sorted((R / 'data/eval').glob('cwmj_m6_confirm_*')):
        source = trial / 'result.json'
        if not source.exists():
            continue
        try:
            result = json.loads(source.read_text())
        except json.JSONDecodeError:
            continue
        if result.get('status') != 'completed' or any(q['past_contact'] for q in result['requests']):
            continue
        name = trial.name.removeprefix('cwmj_m6_confirm_')
        targets = [R / 'data/reports/cwmj_m6_confirmation' / name, DEST / name]
        if not all((p / 'report.md').exists() for p in targets):
            continue
        if all((p / 'caption_correction.json').exists() for p in targets):
            continue
        with h5py.File(trial / 'rollout.hdf5') as h:
            n = len(h['state_14'])
            if not n:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout='constrained')
            for ax, i, label in zip(axes, [0, n-1, n-1],
                                    ['Initial', 'No past-contact request; final frame', 'Last captured pre-action']):
                ax.imshow(h['images/head'][i]); ax.set_title(f'{label} ({i})', fontsize=8); ax.axis('off')
            for target in targets:
                image = target / 'camera_frames.png'
                legacy = target / 'camera_frames_original_caption.png'
                if not legacy.exists():
                    shutil.copy2(image, legacy)
                fig.savefig(image, dpi=160)
                report = target / 'report.md'
                note = '\n\n图注勘误：本回合没有过去接触请求。旧生成器在缺少接触时取最后帧，却误标“首次接触”；现已修正中图标题。原图保留为 camera_frames_original_caption.png。原始轨迹、模型、成功判据与实验代码均未改变。\n'
                text = report.read_text()
                if '图注勘误：' not in text:
                    report.write_text(text + note)
                evidence = dict(reason='no past-contact request; fallback final frame mislabeled',
                                result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                original_image_sha256=hashlib.sha256(legacy.read_bytes()).hexdigest(),
                                corrected_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                                correction_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                (target / 'caption_correction.json').write_text(json.dumps(evidence, indent=2))
            plt.close(fig)
        corrected.append(name)
    print(json.dumps({'corrected_reports': corrected}))


if __name__ == '__main__':
    correct()
