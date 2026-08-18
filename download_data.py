"""Download benchmark data for the SLAM project.

KITTI odometry is an industry-standard outdoor driving benchmark. TUM RGB-D
fr1/desk is a smaller indoor sequence that is easier to download and useful for
testing texture-rich room-scale motion, even though this project uses only RGB.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm


URLS = {
    "kitti_gray": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip",
    "kitti_calib": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip",
    "kitti_poses": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip",
    "tum_desk": "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
}


def download_with_progress(url: str, output: Path) -> None:
    """Download a URL with progress, speed, and estimated remaining time."""
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "slam-project-downloader"})
    with urllib.request.urlopen(request) as response:
        total = int(response.headers.get("Content-Length", "0"))
        start = time.time()
        with output.open("wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=output.name) as bar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                bar.update(len(chunk))
                elapsed = max(time.time() - start, 1e-6)
                speed = bar.n / elapsed
                remaining = (total - bar.n) / speed if total and speed > 0 else 0
                bar.set_postfix_str(f"{speed/1e6:.2f} MB/s, ETA {remaining/60:.1f} min")
    if total and output.stat().st_size < total * 0.99:
        raise RuntimeError(f"Download looks incomplete: {output}")


def extract_archive(archive: Path, destination: Path) -> None:
    """Extract zip or tgz archives into destination."""
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(destination)
    elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive) as tf:
            tf.extractall(destination)
    else:
        raise ValueError(f"Unknown archive type: {archive}")


def copy_sample_frames(data_root: Path, sample_dir: Path, limit: int = 50) -> int:
    """Copy up to limit RGB/grayscale frames into data/sample for instant tests."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(data_root.rglob("*.png")) + sorted(data_root.rglob("*.jpg"))
    count = 0
    for image in images[:limit]:
        shutil.copy2(image, sample_dir / f"{count:06d}{image.suffix.lower()}")
        count += 1
    return count


def download_tum(data_dir: Path) -> None:
    """Download and extract the smaller TUM fr1/desk dataset."""
    tum_dir = data_dir / "tum"
    archive = tum_dir / "rgbd_dataset_freiburg1_desk.tgz"
    if not archive.exists():
        download_with_progress(URLS["tum_desk"], archive)
    extract_archive(archive, tum_dir)
    copied = copy_sample_frames(tum_dir, data_dir / "sample")
    print(f"Downloaded TUM fr1/desk to {tum_dir}")
    print(f"Created {copied} sample frames in {data_dir / 'sample'}")


def download_kitti(data_dir: Path) -> None:
    """Download KITTI odometry archives.

    Official KITTI gray images are distributed as a large archive containing all
    odometry sequences, so this script downloads the official archive and then
    creates a sequence-00 sample view when extraction finishes.
    """
    kitti_dir = data_dir / "kitti"
    for name in ["kitti_gray", "kitti_calib", "kitti_poses"]:
        archive = kitti_dir / Path(URLS[name]).name
        if not archive.exists():
            download_with_progress(URLS[name], archive)
        extract_archive(archive, kitti_dir)
    copied = copy_sample_frames(kitti_dir / "dataset" / "sequences" / "00", data_dir / "sample")
    print(f"Downloaded KITTI odometry archives to {kitti_dir}")
    print(f"Created {copied} sequence-00 sample frames in {data_dir / 'sample'}")


def main() -> None:
    """CLI for choosing KITTI, TUM, or both datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tum", "kitti", "both"], default="tum")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if args.dataset in ("tum", "both"):
        download_tum(data_dir)
    if args.dataset in ("kitti", "both"):
        download_kitti(data_dir)
    print("Download summary:")
    print(f"- Data root: {data_dir.resolve()}")
    print(f"- Sample frames: {(data_dir / 'sample').resolve()}")


if __name__ == "__main__":
    main()
