"""
inference/predict_video.py
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.videomae_finetune import build_model
from utils.config_loader import load_config


UCF_CRIME_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "NormalVideos", "RoadAccidents",
    "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


def video_to_clips(video_path, num_frames=16, clip_stride=8, image_size=224):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size))
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    cap.release()

    if len(frames) < num_frames:
        raise ValueError(f"Video too short: {len(frames)} frames < {num_frames} needed.")

    clips = []
    start = 0
    while start + num_frames <= len(frames):
        clips.append(torch.stack(frames[start:start + num_frames], dim=0))
        start += clip_stride

    print(f"  Video  : {Path(video_path).name}")
    print(f"  Frames : {len(frames)}  |  FPS: {fps:.1f}  |  Duration: {len(frames)/fps:.1f}s")
    print(f"  Clips  : {len(clips)}")
    return clips, fps


@torch.no_grad()
def predict_clips(model, clips, device, class_names, top_k=3, batch_size=4):
    model.eval()
    results = []
    for b in range(0, len(clips), batch_size):
        batch = torch.stack(clips[b:b + batch_size], dim=0).to(device)
        probs = F.softmax(model(batch), dim=-1)
        for i, prob in enumerate(probs):
            top_vals, top_idxs = prob.topk(min(top_k, len(class_names)))
            results.append({
                "clip_idx"  : b + i,
                "top1_class": class_names[top_idxs[0].item()],
                "top1_conf" : round(top_vals[0].item(), 4),
                "top_k"     : [{"class": class_names[idx.item()],
                                 "conf": round(val.item(), 4)}
                                for val, idx in zip(top_vals, top_idxs)],
            })
    return results


def summarize(clip_results, fps, num_frames, clip_stride):
    HIGH_RISK = {"Abuse","Arson","Assault","Explosion","Fighting","Robbery","Shooting"}
    votes = [r["top1_class"] for r in clip_results]
    vote_count = Counter(votes)
    final_cls = vote_count.most_common(1)[0][0]
    confs = [r["top1_conf"] for r in clip_results if r["top1_class"] == final_cls]
    return {
        "final_prediction" : final_cls,
        "confidence"       : round(sum(confs) / len(confs), 4),
        "high_risk"        : final_cls in HIGH_RISK,
        "vote_distribution": dict(vote_count),
        "total_clips"      : len(clip_results),
    }


def predict(video_path, checkpoint, config_path="training/config.yaml",
            top_k=3, save_json=None):
    cfg = load_config(config_path)
    device_str = cfg["project"]["device"]
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else \
                     "mps"  if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)

    num_frames  = cfg["data"]["num_frames"]
    image_size  = cfg["data"]["image_size"]
    num_classes = cfg["data"]["num_classes"]
    class_names = UCF_CRIME_CLASSES[:num_classes]

    print("\n── SentinelMAE Inference ──────────────────────────────")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Device     : {device}")

    cfg["project"]["device"] = device_str
    model = build_model(cfg)
    ckpt  = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(device)
    print("  Model      : loaded ✓")

    t0 = time.time()
    clips, fps = video_to_clips(video_path, num_frames, 8, image_size)
    clip_results = predict_clips(model, clips, device, class_names, top_k)
    elapsed = time.time() - t0

    summary = summarize(clip_results, fps, num_frames, 8)

    print("\n── Per-clip predictions ───────────────────────────────")
    for r in clip_results:
        t_start = r["clip_idx"] * 8 / fps
        t_end   = t_start + num_frames / fps
        top_str = "  |  ".join(f"{x['class']} {x['conf']*100:.1f}%" for x in r["top_k"])
        print(f"  [{t_start:6.2f}s – {t_end:6.2f}s]  {top_str}")

    print("\n── Video-level Summary ────────────────────────────────")
    print(f"  Prediction : {summary['final_prediction']}")
    print(f"  Confidence : {summary['confidence']*100:.1f}%")
    print(f"  High Risk  : {'⚠ YES' if summary['high_risk'] else 'NO'}")
    print(f"  Votes      : {summary['vote_distribution']}")
    print(f"  Time       : {elapsed:.2f}s")
    print("───────────────────────────────────────────────────────\n")

    output = {"summary": summary, "clips": clip_results}
    if save_json:
        with open(save_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved → {save_json}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="training/config.yaml")
    parser.add_argument("--top_k",      type=int, default=3)
    parser.add_argument("--save_json",  default=None)
    args = parser.parse_args()
    predict(args.video, args.checkpoint, args.config, args.top_k, args.save_json)