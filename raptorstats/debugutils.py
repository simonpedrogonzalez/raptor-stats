import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.plot import plotting_extent
from rasterio.windows import transform as window_transform
from rasterstats.io import Raster


def plot_mask_comparison(
    global_mask: np.ndarray,
    ref_mask: np.ndarray,
    features: gpd.GeoDataFrame,
    transform: Affine,
    window: tuple = None,
    title: str = "Mask comparison",
    show_grid: bool = True,
    scanlines: np.ndarray = None,
    idx: int = None,
    used_nodes: list = None,
):
    """
    Visualise agreement / disagreement between two binary-or-indexed masks.

    comparison code
        0 → no mask
        1 → equal & non-zero  (gray)
        2 → only in global    (blue)
        3 → only in ref       (red)
    """

    # --- 1. build comparison image -----------------------------------------
    comparison = np.zeros_like(global_mask, dtype=np.uint8)
    comp_1 = (global_mask == ref_mask) & (global_mask != 0)
    comp_2 = (global_mask != 0) & (ref_mask == 0)
    comp_3 = (global_mask == 0) & (ref_mask != 0)
    comparison[comp_1] = 1
    comparison[comp_2] = 2
    comparison[comp_3] = 3

    # nothing to show?
    if comparison.max() == 0:
        print("⚠  comparison image is empty (all zeros)")
    # -----------------------------------------------------------------------

    cmap = ListedColormap(["white", "gray", "blue", "red"])
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    subset_transform = window_transform(window, transform)
    extent = plotting_extent(global_mask, transform=subset_transform)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title)
    ax.imshow(comparison, cmap=cmap, norm=norm, origin="upper", extent=extent)

    # ---- 2. optional pixel grid in world coords ---------------------------
    if show_grid:
        h, w = global_mask.shape
        # row boundaries (world Y)
        row_edges = np.arange(h + 1)
        col_edges = np.arange(w + 1)

        # pixel centre (0,0) is at col=0.5, row=0.5 => boundaries at integers
        # Convert to world: (x, y) = transform * (col, row)
        for r in row_edges:
            _, y = subset_transform * (0, r)
            ax.axhline(y, color="black", linewidth=0.2, alpha=0.3)
        for c in col_edges:
            x, _ = subset_transform * (c, 0)
            ax.axvline(x, color="black", linewidth=0.2, alpha=0.3)
    # ---------------------- grid lines -------------------------------------

    # ---- 3. overlay geometries -------------------------------------------
    features.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)
    # -----------------------------------------------------------------------

    if scanlines is not None:
        for y in scanlines:
            ax.axhline(y, color="black", linewidth=0.5, linestyle="--")
    # legend
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(facecolor="gray", edgecolor="none", label="match"),
        Patch(facecolor="blue", edgecolor="none", label="global only"),
        Patch(facecolor="red", edgecolor="none", label="ref only"),
    ]
    ax.legend(handles=legend_patches, loc="upper right")

    if idx is not None:
        all_nodes = list(idx.intersection(idx.bounds, objects=True))
        all_boxes = [node.object.box for node in all_nodes]
        gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(all_boxes), crs=features.crs)
        gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5)

    if used_nodes is not None:
        used_boxes = [node.box for node in used_nodes]
        gdf_used = gpd.GeoDataFrame(geometry=gpd.GeoSeries(used_boxes), crs=features.crs)
        gdf_used.plot(ax=ax, facecolor="green", edgecolor="black", linewidth=0.5, alpha=0.6)

    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


import pprint
from textwrap import indent

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterstats import zonal_stats

from raptorstats.raster_methods import Masking  # <- class you showed


def compare_stats(
    mystats,  # scanline stats
    raster_path: str,
    vector_path: str,
    stats: list | str = ("min", "max", "mean", "count"),
    show_diff: bool = True,
    precision: int = 5,
):
    """
    Run three zonal-stat engines and print results side-by-side.

    Parameters
    ----------
    raster_path : str
    vector_path : str
    stats       : list[str] | str   statistics to compute (like in rasterstats)
    show_diff   : bool              whether to print per-stat numeric diffs
    precision   : int               rounding / diff precision
    """

    # --- load vector layer once -------------------------------------------
    gdf = gpd.read_file(vector_path) if isinstance(vector_path, str) else vector_path

    scan_res = mystats

    # --- open raster & run methods ----------------------------------------
    with rio.open(raster_path) as ds:
        # 1. your Scanline

        # 3. rasterstats (all_touched=True like your Masking wrapper)
        rs_res = zonal_stats(vector_path, raster_path, stats=stats.requested)

    # --- pretty print ------------------------------------------------------
    pp = pprint.PrettyPrinter(compact=True, sort_dicts=True)
    all_diffs = []
    print("\n=== Zonal-stat comparison ===")
    for i, (s, r) in enumerate(zip(scan_res, rs_res), start=1):
        print(f"\nFeature #{i}")
        print("  Scanline : ", end="")
        pp.pprint(s)
        # print("  Masking  : ", end=""); pp.pprint(m)
        print("  RasterSt : ", end="")
        pp.pprint(r)

        if show_diff:
            diff = {}
            for k in set(s) & set(r):
                sv = round(s.get(k, float("nan")), precision)
                rv = round(r.get(k, float("nan")), precision)
                # show diff vs rasterstats (change to scanline vs masking if you prefer)
                if all(isinstance(v, (int, float)) for v in (sv, rv)):
                    diff[k] = round(sv - rv, precision)
            print("  Δ(Scanline-RasterSt):", indent(pp.pformat(diff), 2 * " "))
            if diff['count'] > 0:
                all_diffs.append({
                    "feature": i-1,
                    "count1": s.get("count", 0),
                    "count2": r.get("count", 0),
                })

    print("\nDone.\n")
    return all_diffs


import numpy as np
import rasterio as rio
from rasterio.windows import transform as window_transform
from rasterstats.utils import rasterize_geom


def compute_stats(masked: np.ndarray):
    """Computes the statistics for the masked array.

    The code is based on the rasterstats library, so no difference in performance
    should be expected in this part.
    """

    stats = {"min", "max", "mean", "count", "sum"}
    stats_out = {}

    if masked.compressed().size == 0:
        for stat in self.stats:
            if stat == "count":
                stats_out[stat] = 0
            else:
                stats_out[stat] = None
        return stats_out

    # We are prly only use the mean but just in case
    if "min" in stats:
        stats_out["min"] = float(masked.min())
    if "max" in stats:
        stats_out["max"] = float(masked.max())
    if "mean" in stats:
        stats_out["mean"] = float(masked.mean())
    if "count" in stats:
        stats_out["count"] = int(masked.count())
    if "sum" in stats:
        stats_out["sum"] = float(masked.sum())

    return stats_out


def ref_mask_rasterstats(
    features,
    raster: rio.DatasetReader,
    window: rio.windows.Window,
    all_touched: bool = False,
):

    # ---------------- window geometry & output array ----------------------
    win = window  # .round_offsets().round_lengths()
    sub_transform = window_transform(win, raster.transform)
    out_h = int(win.height)
    out_w = int(win.width)
    ref_mask = np.zeros((out_h, out_w), dtype=np.uint16)  # big enough for 65k features

    # ---------------- iterate over features -------------------------------
    # NOTE: rasterstats loops feature-by-feature; later features overwrite earlier ones.

    affine = raster.transform
    nodata = raster.nodata
    band = 1  # rasterstats uses band 1 by default
    with Raster(raster.name, affine, nodata, band) as rast:
        for idx, geom in enumerate(features.geometry):
            if geom is None or geom.is_empty:
                continue
            # geom as shapely object
            # geom = mapping(geom)

            # Rasterstats uses a boolean mask from rasterize_geom:
            geom_mask = rasterize_geom(
                # mapping(geom),
                geom=geom,  # geojson geometry
                like=type(
                    "Like",
                    (),
                    {  # fake "like" object with .shape & .affine
                        "shape": (out_h, out_w),
                        "affine": sub_transform,
                    },
                )(),
                all_touched=all_touched,
            )

            # burn feature id (idx+1) where mask is True
            ref_mask[geom_mask] = idx + 1

            fsrc = raster.read(window=win)
            has_nan = np.issubdtype(fsrc.dtype, np.floating) and np.isnan(fsrc.min())
            isnodata = fsrc == raster.nodata
            if has_nan:
                isnodata = isnodata | np.isnan(fsrc)
            masked = np.ma.MaskedArray(fsrc, mask=(isnodata | ~geom_mask))
            stats = compute_stats(masked)
            print(f"Feature #{idx+1} stats: {stats['mean']}")

    return ref_mask
