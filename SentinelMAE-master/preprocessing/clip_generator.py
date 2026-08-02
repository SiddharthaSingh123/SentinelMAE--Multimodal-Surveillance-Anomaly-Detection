"""
preprocessing/clip_generator.py
Handles PNG frame folders (UCF-Crime format).
Each class folder contains frames named: VideoName_frameNumber.png
"""

import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def build_dataset(raw_dir="data/raw", out_dir="data/processed",
                  num_frames=16, image_size=224,
                  val_split=0.2, seed=42, max_clips_per_video=10):

    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    random.seed(seed)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    class_dirs = sorted([d for d in raw_path.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    total = {"train": 0, "val": 0}

    for class_dir in class_dirs:
        # Group frames by video name
        # Frame format: Fighting029_x264_1110.png → video = Fighting029_x264
        all_frames = sorted(class_dir.glob("*.png"))
        if not all_frames:
            print(f"  [WARN] No frames in {class_dir.name}, skipping.")
            continue

        # Group by video prefix (everything except last _number)
        videos = defaultdict(list)
        for f in all_frames:
            parts = f.stem.rsplit("_", 1)
            video_name = parts[0]
            frame_num  = int(parts[1])
            videos[video_name].append((frame_num, f))

        # Sort frames within each video
        for v in videos:
            videos[v].sort(key=lambda x: x[0])

        video_names = list(videos.keys())
        random.shuffle(video_names)
        n_val = max(1, int(len(video_names) * val_split))
        splits = {"val": video_names[:n_val], "train": video_names[n_val:]}

        print(f"\nClass '{class_dir.name}': {len(splits['train'])} train, {len(splits['val'])} val videos")

        for split, vid_list in splits.items():
            out_class = out_path / split / class_dir.name
            out_class.mkdir(parents=True, exist_ok=True)

            for video_name in tqdm(vid_list, desc=f"  [{split}] {class_dir.name}"):
                frames = [f for _, f in videos[video_name]]

                if len(frames) < num_frames:
                    continue

                # Sliding window clips
                clips = []
                stride = max(1, len(frames) // max_clips_per_video)
                start  = 0
                while start + num_frames <= len(frames) and len(clips) < max_clips_per_video:
                    window = frames[start:start + num_frames]
                    tensors = [transform(Image.open(f).convert("RGB")) for f in window]
                    clip = torch.stack(tensors, dim=0)  # (T, C, H, W)
                    clips.append(clip)
                    start += stride

                for i, clip in enumerate(clips):
                    torch.save(clip, out_class / f"{video_name}_clip{i:03d}.pt")
                total[split] += len(clips)

    print(f"\n Done!  Train clips: {total['train']}  |  Val clips: {total['val']}")


if __name__ == "__main__":
    build_dataset()