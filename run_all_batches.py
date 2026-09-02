"""Runs tree_extraction over every image folder under ROOT_DIR, one subprocess
each. Each folder gets its own output subfolder and CSV, same as running the
script directly."""

import os
import sys
import glob
import time
import subprocess

SCRIPT   = 'tree_extraction_v162_parallel'      
ROOT_DIR = os.path.join('data', 'batches')
OUT_NAME = 'extracted_batch_v162'                           # subfolder created inside each image folder
IMG_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp')


def find_image_folders(root):
    """First folder on each branch that holds images directly. Handles both
    layouts: Batch 1 keeps them in 'Sample N', Batch 2 in 'images'. Stops
    descending once found, so 'Selected', 'test', '_processed' etc. inside a
    sample are ignored."""
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.lower().startswith('extracted')]
        if any(glob.glob(os.path.join(dirpath, e)) for e in IMG_EXTS):
            found.append(dirpath)
            dirnames[:] = []          # don't descend into a sample's subfolders
    return sorted(found)

    
def main():
    folders = find_image_folders(ROOT_DIR)
    if not folders:
        raise SystemExit(f"No image folders under {os.path.abspath(ROOT_DIR)}")

    print(f"{len(folders)} folder(s) to process:")
    for f in folders:
        print(f"   {os.path.relpath(f, ROOT_DIR)}")
    print()

    t0 = time.perf_counter()
    done, failed = [], []

    for i, folder in enumerate(folders, 1):
        rel = os.path.relpath(folder, ROOT_DIR)
        out_dir = os.path.join(folder, OUT_NAME)
        print(f"\n{'='*70}\n[{i}/{len(folders)}] {rel}\n{'='*70}")

        env = dict(os.environ, TREE_IN_DIR=folder, TREE_OUT_DIR=out_dir)
        result = subprocess.run([sys.executable, SCRIPT], env=env)

        (done if result.returncode == 0 else failed).append(rel)
        if result.returncode != 0:
            print(f"  !! exited {result.returncode}, continuing")

    print(f"\n{len(done)} ok, {len(failed)} failed, "
          f"{(time.perf_counter()-t0)/60:.1f} min total")
    for f in failed:
        print(f"  failed: {f}")


if __name__ == '__main__':
    main()