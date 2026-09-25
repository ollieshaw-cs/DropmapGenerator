"""
Batch-converts a directory of .obj terrain chunks into cached binary
.ply files (same name, .ply extension) so future loads skip the slow
text parsing that trimesh has to do for OBJ files.

Run this once whenever the source OBJs change. After that, point
loadTerrainMeshes() at the same directory and it'll pick up the .ply
cache instead (see the snippet at the bottom of this file).
"""

import time
from pathlib import Path

import trimesh


def convertObjDirectoryToCache(
    directory: Path,
    overwrite: bool = False
):
    """
    Convert every .obj in `directory` to a binary .ply next to it.
    Skips files that already have an up-to-date .ply unless
    overwrite=True.
    """

    obj_files = sorted(directory.glob("*.obj"))

    if not obj_files:
        raise RuntimeError(f"No OBJ files found in {directory}")

    print(f"Found {len(obj_files)} OBJ files in {directory}")
    print()

    total_start = time.time()

    for index, obj_file in enumerate(obj_files, start=1):

        ply_file = obj_file.with_suffix(".ply")

        if ply_file.exists() and not overwrite:
            print(f"[{index}/{len(obj_files)}] {obj_file.name} -> "
                  f"{ply_file.name} (already cached, skipping)")
            continue

        print(f"[{index}/{len(obj_files)}] {obj_file.name} ...", end="", flush=True)

        start = time.time()

        mesh = trimesh.load(obj_file, process=False)

        # Some OBJs load as a Scene (multiple sub-geometries) rather
        # than a single Trimesh, same as in the heightmap script.
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                tuple(
                    geometry
                    for geometry in mesh.geometry.values()
                    if isinstance(geometry, trimesh.Trimesh)
                )
            )

        mesh.export(ply_file, file_type="ply", encoding="binary")

        elapsed = time.time() - start

        print(
            f" done in {elapsed:.1f}s "
            f"({len(mesh.vertices):,} verts, {len(mesh.faces):,} tris) "
            f"-> {ply_file.stat().st_size / 1_048_576:.1f} MB"
        )

    total_elapsed = time.time() - total_start
    print()
    print(f"All done in {total_elapsed:.1f}s")


if __name__ == "__main__":

    terrainOBJs = Path(r"E:\FortniteTerrain\Rows")

    convertObjDirectoryToCache(
        directory=terrainOBJs,
        overwrite=False  # set True to force re-convert everything
    )