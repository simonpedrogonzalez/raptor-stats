import geopandas as gpd
import rasterio as rio

# --- make sure your Scanline & helpers are importable ---
from raptorstats.scanline import Scanline     # ← adjust import path
from rasterstats.utils import check_stats      # re-use existing util for VALID_STATS, etc.


def zonal_stats(
        vectors,
        raster,
        stats=None,
        layer=0,
        band=1,
        nodata=None,
        affine=None,
        geojson_out=False,
        prefix=None,
        **kwargs,
):
    """
    Zonal statistics using the Scanline rasterisation method.

    Parameters
    ----------
    vectors :  str | GeoDataFrame | anything GeoPandas can read
    raster  :  str | path-like | rasterio.DatasetReader
    stats   :  list[str]            statistics to compute (min, max, mean, etc.)
    layer   :  int | str            layer name/number for multi-layer vector sources
    band    :  int                  raster band (1-based)
    nodata  :  float | None         overrides raster NODATA value
    affine  :  affine.Affine | None required if `raster` is a NumPy array (not used here)
    geojson_out : bool              if True, return GeoJSON-like Features
    prefix  :  str | None           prefix every stat key (useful for merges)

    Returns
    -------
    list[dict]                      one stats-dict (or Feature) per input geometry
    """

    # ---- 1. prepare stats list (reuse rasterstats helper) ----
    stats, _ = check_stats(stats, categorical=False)   # Scanline = numeric only

    # ---- 2. read vectors (GeoPandas) ----
    if isinstance(vectors, gpd.GeoDataFrame):
        gdf = vectors
    else:
        gdf = gpd.read_file(vectors, layer=layer)
        file_path = vectors
        gdf.attrs["file_path"] = file_path  # store original file path in GeoDataFrame attrs

    # ---- 3. open raster once ----
    if isinstance(raster, rio.io.DatasetReaderBase):
        ds = raster
        must_close = False
    else:
        ds = rio.open(raster, "r")
        must_close = True

    try:
        # ---- 4. run Scanline ----
        sl = Scanline()
        results = sl(ds, gdf, stats=stats)   # Scanline.__call__ returns list[dict]

        # ---- 5. attach / post-process ----
        if prefix:
            results = [
                {f"{prefix}{k}": v for k, v in res.items()} if res is not None else None
                for res in results
            ]

        if geojson_out:
            features_out = []
            for feat, res in zip(gdf.itertuples(index=False), results):
                f = {
                    "type": "Feature",
                    "geometry": feat.geometry.__geo_interface__,
                    "properties": res or {},
                }
                features_out.append(f)
            return features_out
        else:
            return results

    finally:
        if must_close:
            ds.close()
