"""
Generate a heightmap from a minimap image by raycasting down onto
terrain PLY chunks.

Memory-safe version: processes ONE terrain chunk fully (load -> raycast
every pixel batch -> discard) before moving to the next.
"""

import gc

import numpy as np
import trimesh
from PIL import Image
from pathlib import Path


# ---------------------------------------------------------------------
# Minimap -> Blender coordinate calibration
# ---------------------------------------------------------------------

locationMappings = {
    (1215, 477): (367.040, 796.703),
    (389, 1412): (-844.465, -573.104),
    (501, 462): (-680.836, 818.871),
}

minimap = np.array(list(locationMappings.keys()), dtype=float)
blender = np.array(list(locationMappings.values()), dtype=float)

# Fit:
#
# blender_x = xScale * minimap_x + xOffset
# blender_z = yScale * minimap_y + yOffset
#
# np.polyfit(..., 1) performs a least-squares linear fit.

xScale, xOffset = np.polyfit(
    minimap[:, 0],
    blender[:, 0],
    1
)

yScale, yOffset = np.polyfit(
    minimap[:, 1],
    blender[:, 1],
    1
)

print("Coordinate mapping:")
print(f"  xScale:  {xScale:.9f}")
print(f"  xOffset: {xOffset:.9f}")
print(f"  yScale:  {yScale:.9f}")
print(f"  yOffset: {yOffset:.9f}")


# ---------------------------------------------------------------------
# Optional additional offsets
# ---------------------------------------------------------------------

X_OFFSET = -1.4
Z_OFFSET = -0.5


def minimapToBlender(coords) -> tuple[float, float]:
    """
    Convert minimap pixel coordinates to Blender X/Z coordinates.

    Uses the least-squares calibration generated from locationMappings.
    """

    x, y = coords

    blender_x = x * xScale + xOffset
    blender_z = y * yScale + yOffset

    blender_x += X_OFFSET
    blender_z += Z_OFFSET

    return blender_x, blender_z

def blenderToMinimap(coords) -> tuple[float, float]:
    
    blender_x, blender_z = coords

    blender_x -= X_OFFSET
    blender_z -= Z_OFFSET

    minimap_x = (blender_x - xOffset) / xScale
    minimap_y = (blender_z - yOffset) / yScale

    return minimap_x, minimap_y


def loadPlyMesh(ply_file: Path):
    """
    Load a single binary PLY terrain chunk as a Trimesh.
    """

    mesh = trimesh.load(ply_file, process=False)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(
                geometry
                for geometry in mesh.geometry.values()
                if isinstance(geometry, trimesh.Trimesh)
            )
        )

    return mesh


def findPlyFiles(directory: Path):
    """
    Find all PLY terrain chunks.
    """

    ply_files = sorted(directory.glob("*.ply"))

    if not ply_files:
        raise RuntimeError(f"No PLY files found in {directory}")

    return ply_files


def computeHighestY(ply_files):
    """
    First lightweight pass: load each chunk just long enough to read its
    bounding box, then discard it.

    This avoids ever holding more than one mesh in memory.
    """

    highest = -np.inf

    print("Scanning chunks for max terrain height...")

    for ply_file in ply_files:
        mesh = loadPlyMesh(ply_file)

        highest = max(
            highest,
            float(mesh.bounds[1][1])
        )

        del mesh
        gc.collect()

    print(f"Highest terrain Y: {highest:.3f}")

    return highest


def generateHeightMap(
    image_path: str,
    terrain_directory: Path,
    output_path: str = "heightmap.npy",
    batch_size: int = 100_000,
    highest_y: float | None = None,
):
    """
    Generate a heightmap from the minimap.

    The resulting array has:

        heightmap[yPixel, xPixel] = Blender Y height

    Chunks are processed one at a time so that only one mesh and its
    ray-intersection data are resident in memory at once.
    """

    # ---------------------------------------------------------
    # Load minimap
    # ---------------------------------------------------------

    image = Image.open(image_path)

    width, height = image.size

    print(f"Minimap size: {width} x {height}")

    heightmap = np.full(
        (height, width),
        np.nan,
        dtype=np.float32
    )

    # ---------------------------------------------------------
    # Generate all minimap pixel coordinates
    # ---------------------------------------------------------

    pixel_x, pixel_y = np.meshgrid(
        np.arange(width),
        np.arange(height)
    )

    pixel_x = pixel_x.ravel()
    pixel_y = pixel_y.ravel()

    total_pixels = len(pixel_x)

    # ---------------------------------------------------------
    # Find chunks and determine ray-start height
    # ---------------------------------------------------------

    ply_files = findPlyFiles(terrain_directory)

    print(f"Found {len(ply_files)} PLY terrain chunks")

    if highest_y is None:
        highest_y = computeHighestY(ply_files)

    # ---------------------------------------------------------
    # Process one chunk at a time
    # ---------------------------------------------------------

    for chunk_index, ply_file in enumerate(
        ply_files,
        start=1
    ):

        print(
            f"[{chunk_index}/{len(ply_files)}] "
            f"Loading {ply_file.name}..."
        )

        mesh = loadPlyMesh(ply_file)

        print(
            f"  vertices: {len(mesh.vertices):,} | "
            f"triangles: {len(mesh.faces):,}"
        )

        # -----------------------------------------------------
        # Process minimap pixels in batches
        # -----------------------------------------------------

        for batch_start in range(
            0,
            total_pixels,
            batch_size
        ):

            batch_end = min(
                batch_start + batch_size,
                total_pixels
            )

            px = pixel_x[batch_start:batch_end]
            py = pixel_y[batch_start:batch_end]

            # -------------------------------------------------
            # Convert minimap coordinates -> Blender X/Z
            # -------------------------------------------------

            # New least-squares mapping:
            #
            # Blender X = xScale * minimap X + xOffset
            # Blender Z = yScale * minimap Y + yOffset

            blender_x = (
                px * xScale
                + xOffset
                + X_OFFSET
            )

            blender_z = (
                py * yScale
                + yOffset
                + Z_OFFSET
            )

            # -------------------------------------------------
            # Rays start above the highest terrain point and
            # shoot straight down along Blender Y.
            # -------------------------------------------------

            origins = np.column_stack([
                blender_x,
                np.full(
                    len(blender_x),
                    highest_y + 100.0
                ),
                blender_z
            ])

            directions = np.zeros_like(origins)

            directions[:, 1] = -1.0

            # -------------------------------------------------
            # Raycast against this chunk only
            # -------------------------------------------------

            locations, ray_indices, _ = (
                mesh.ray.intersects_location(
                    ray_origins=origins,
                    ray_directions=directions,
                    multiple_hits=False
                )
            )

            if len(locations) == 0:
                continue

            hit_heights = locations[:, 1].astype(
                np.float32
            )

            hit_px = px[ray_indices]
            hit_py = py[ray_indices]

            # -------------------------------------------------
            # Keep the highest hit seen so far across ALL
            # terrain chunks.
            # -------------------------------------------------

            current = heightmap[
                hit_py,
                hit_px
            ]

            keep = (
                np.isnan(current)
                | (hit_heights > current)
            )

            heightmap[
                hit_py[keep],
                hit_px[keep]
            ] = hit_heights[keep]

            print(
                f"  pixels "
                f"{batch_start:,}-{batch_end:,} / "
                f"{total_pixels:,}",
                end="\r"
            )

        print()

        # -----------------------------------------------------
        # Release this chunk and its ray-intersection tree
        # before loading the next chunk.
        # -----------------------------------------------------

        del mesh
        gc.collect()

    # ---------------------------------------------------------
    # Save heightmap
    # ---------------------------------------------------------

    np.save(
        output_path,
        heightmap
    )

    print()
    print(f"Saved heightmap to: {output_path}")
    print(f"Shape: {heightmap.shape}")

    valid = heightmap[
        ~np.isnan(heightmap)
    ]

    if len(valid):
        print(
            f"Minimum height: {valid.min():.3f}"
        )

        print(
            f"Maximum height: {valid.max():.3f}"
        )

    return heightmap


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    terrainDirectory = Path(
        r"E:\FortniteTerrain\Rows"
    )

    heightmap = generateHeightMap(
        image_path="assets/images/Minimap.png",
        terrain_directory=terrainDirectory,
        output_path="data/heightmap.npy"
    )