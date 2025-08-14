import geopandas as gpd
import line_profiler
import numpy as np
import rasterio as rio
from rasterio.features import geometry_window

from raptorstats.stats import Stats


class ZonalStatMethod:

    __name__ = "ZonalStatMethod"

    def __init__(self):
        pass

    def _get_params(self):
        return {}

    def _precomputations(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
        """Precomputations to be done before the main loop, for example for creating
        indexes, QuadTrees, etc. Implementing this method is optional.

        Parameters
        ----------
        features : gpd.GeoDataFrame
        raster : rio.DatasetReader

        """
        pass

    def _compute_stats(
        self,
        feature: gpd.GeoDataFrame,
        raster: rio.DatasetReader,
        window: rio.windows.Window,
    ):
        """Method to be implemented by the subclass to compute the statistics of a
        single geometry."""
        raise NotImplementedError

    def _run(self, vector_layer: gpd.GeoDataFrame, raster: rio.DatasetReader):
        results = []
        self._precomputations(vector_layer, raster)
        for geom in vector_layer.geometry:
            try:
                window = geometry_window(raster, [geom])
            except rio.errors.WindowError:
                # there is no intersection
                results.append(None)
                continue
            feature = gpd.GeoDataFrame(geometry=[geom], crs=vector_layer.crs)
            results.append(self._compute_stats(feature, raster, window))
        return results

    def __call__(
        self, raster: rio.DatasetReader, features: gpd.GeoDataFrame, stats: Stats
    ):
        """User-facing method to compute the zonal statistics.

        Parameters
        ----------
        raster : rio.DatasetReader
        features : gpd.GeoDataFrame
        stats : Stats

        Returns
        -------
        List[Dict[str, float]]

        """
        self.stats = stats
        self.raster = raster
        self.features = features

        results = self._run(self.features, self.raster)
        return results
