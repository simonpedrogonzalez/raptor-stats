import warnings

from raptorstats.agqt import AggQuadTree
from raptorstats.io import (
    open_raster,
    open_vector,
    validate_is_north_up,
    validate_raster_vector_compatibility,
)
from raptorstats.scanline import Scanline
from raptorstats.stats import Stats


def zonal_stats(
    vectors,
    raster,
    stats=None,
    layer=0,
    band=1,
    nodata=None,
    affine=None,
    prefix=None,
    categorical=False,
    method="scanline",
    **kwargs,
):
    """Zonal statistics using the AggQuadTree or Scanline methods.

    The method supports quite a few input types for both vectors and rasters, with options
    to select band, layers, specify affine, etc. However, the loading method will make
    assumptions for some of them, so if your input is exotic (or you see errors),
    ideally you might want to transform it yourself to a north-up, one-band, no-shearing
    rasterio DatasetReader and a geopandas closed-polygon-only GeoDataFrame with the
    same CRS as the raster.

    Parameters
    ----------
    vectors :  str | Path | geopandas.GeoDataFrame | shapely.geometry.base.BaseGeometry | Sequence[BaseGeometry] | dict (GeoJSON-like mapping) | anything GeoPandas can read.
        Vector source containing the geometries. Currently only closed polygons are supported (no single points or lines).
    raster  : str | Path | rasterio.io.DatasetReader | np.ndarray | anything rasterio can read
        Raster source containing the data to compute statistics on.
    stats   : list[str]
        Statistics to compute. Default: ["count", "min", "max", "mean"], also supported:
        ["sum", "std", "median", "majority", "minority", "unique", "range", "nodata", "nan"].
    layer   : int | str
        Layer name/number for multi-layer vector sources
    band    : int
        Raster band (default 1)
    nodata  : float | None
        Overrides raster NODATA value
    affine  : affine.Affine | None
        Required if `raster` is a NumPy array
    prefix  : str | None
        Prefix every stat key (useful for merges)
    categorical : bool
        Whether to compute categorical stats (histogram)
    method  : str
        Zonal statistics method to use ('agqt' or 'scanline'). Default 'scanline'.
            - *scanline*: builds one line per raster row, calculates the intersections with the vector layer,
              identifying the pixels corresponding to each vector feature, and computes the statistics. It reads
              the raster in a single pass, row by row, computing all the feature stats.
            
            - *agqt*: builds an AggQuadTree index for the raster, which allows for fast querying of the statistics
              for a grid of rectangular features. During the zonal_stats call, it identifies the rectangular features
              inside the vector geometries and takes the pre-computed stats from the index file, while using the scanline
              method for pixels inside the geometry that are not covered by the rectangular features.
    build_indices : bool
        Whether to force build (overwrite) the AggQuadTree index if it already exists. Default False.
    index_path : str
        Path to the index file (if using agqt), default None. If the file doesn't exist, it will be created.
        If not specified, the program will try to use a default path in the current running dir.
    max_depth : int
        Maximum depth for the AggQuadTree index (if using agqt), default 5. The index building time for the `agqt` method depends
        on the specified tree depth. For example, building a depth-10 quadtree requires significant time
        and memory, as the algorithm needs to compute statistics for about (4^11 - 1) / 3 ≈ 1.4 million
        rectangular features. The depth of the tree needed depends on the size of the expected features.
        A poorly chosen depth (too low, so that leaf nodes don't fit inside the features, or too high,
        making the tree unnecessarily large) can lead to slower times for the `agqt` method compared to
        the `scanline` method.
    Returns
    -------
    list[dict]
        one stats-dict per input feature

    """

    stats_conf = Stats(stats, categorical=categorical)

    if method == "agqt":
        func = AggQuadTree(**kwargs)
    elif method == "scanline":
        func = Scanline(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'agqt' or 'scanline'.")

    with open_raster(raster, affine=affine, nodata=nodata, band=band) as ds:

        gdf = open_vector(vectors, layer=layer, affine=affine)

        validate_is_north_up(ds.transform)
        validate_raster_vector_compatibility(ds, gdf)

        results = func(ds, gdf, stats=stats_conf)

        results = stats_conf.clean_results(results)

        if prefix:
            results = [
                {f"{prefix}{k}": v for k, v in res.items()} if res is not None else None
                for res in results
            ]

        return results


def build_agqt_index(
    raster,
    max_depth,
    stats=None,
    band=1,
    nodata=None,
    affine=None,
    categorical=False,
    index_path=None,
):
    """Build AggQuadTree index files for the given raster file.

    Parameters
    ----------
    raster : str | Path | rasterio.io.DatasetReader | np.ndarray | anything rasterio can read
    max_depth : int
        Maximum depth for the AggQuadTree index. The index building time for the `agqt` method depends
        on the specified tree depth. For example, building a depth-10 quadtree requires significant time
        and memory, as the algorithm needs to compute statistics for about (4^11 - 1) / 3 ≈ 1.4 million
        rectangular features. The depth of the tree needed depends on the size of the expected features.
        A poorly chosen depth (too low, so that leaf nodes don't fit the features, or too high,
        making the tree unnecessarily large) can lead to slower times for the `agqt` method compared to
        the `scanline` method.
    stats : list[str]
        Statistics to compute. Default: ["count", "min", "max", "mean"], also supported:
        ["sum", "std", "median", "majority", "minority", "unique", "range", "nodata", "nan"]. Important: the
        computed stats for the indices must match the later required stats for the computation, when running zonal_stats.
    band : int
        Raster band (1-based).
    nodata : float | None
        Overrides raster NODATA value.
    affine : affine.Affine | None
        Required if `raster` is a NumPy array.
    categorical : bool
        Whether to compute categorical stats (histogram).
    index_path : str | None
        Path to the index file (if using AggQuadTree), default None. If the file doesn't exist, it will be created.
        If not specified, the program will try to use a default path in the current running dir.

    Returns
    -------
    rtree.index.Index
        The AggQuadTree index object (it is already saved to a file).
    """
    stats_conf = Stats(stats, categorical=categorical)

    agqt = AggQuadTree(max_depth=max_depth, index_path=index_path, build_index=True)
    agqt.stats = stats_conf

    with open_raster(raster, affine=affine, nodata=nodata, band=band) as ds:
        validate_is_north_up(ds.transform)
        if agqt._index_files_exist():
            warnings.warn(
                f"Index files {agqt.index_path}.idx and {agqt.index_path}.dat already exist. They will be overwritten.",
            )
            agqt._delete_index_files()
        agqt._compute_quad_tree(ds)

    return agqt.idx  # Return the index object for further use or inspection
