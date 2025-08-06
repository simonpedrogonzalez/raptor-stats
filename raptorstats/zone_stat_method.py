import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.features import geometry_window
import line_profiler
from raptorstats.stats import Stats
    

class ZonalStatMethod():

    __name__ = "ZonalStatMethod"

    def __init__(self):
        pass

    def _get_params(self):
        return {}

    def _precomputations(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
        """Precomputations to be done before the main loop, for example for
        creating indexes, QuadTrees, etc. Implementing this method is optional.

        Parameters
        ----------
        features : gpd.GeoDataFrame
        raster : rio.DatasetReader
        """
        pass
    
    def _compute_stats_from_array(self, data: np.ndarray):
        """Computes the statistics for the array.

        The code is based on the rasterstats library, so no difference in performance
        should be expected in this part.
        """

        stats = self.stats
        stats_out = {}

        if data.size == 0:
            stats_out = { stat: None for stat in stats }
            if "count" in stats:
                stats_out["count"] = 0
            return stats_out
        

        # We are prly only use the mean but just in case
        if "min" in stats:
            stats_out["min"] = float(data.min())
        if "max" in stats:
            stats_out["max"] = float(data.max())
        if "mean" in stats:
            stats_out["mean"] = float(data.mean())
        if "count" in stats:
            stats_out["count"] = int(len(data))
        if "sum" in stats:
            stats_out["sum"] = float(data.sum())
        if "std" in stats:
            stats_out["std"] = float(data.std())
        if "median" in stats:
            stats_out["median"] = float(np.median(data))

        # Categorical:
        # majority
        # minority
        # value: count
        # unique: len(unique values)
        # 

        return stats_out

    def _compute_stats_from_masked_array(self, masked: np.ndarray):
        """Computes the statistics for the masked array.

        The code is based on the rasterstats library, so no difference in performance
        should be expected in this part.
        """

        stats = self.stats
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


    def _combine_stats(self, results: list):
        """Combine the statistics from a List[Dict[str, float]] into a single dict.

        Useful for integrating partial results.

        Parameters
        ----------
        stats : List[Dict[str, float]]
            List of statistics for different regions of the same geometry.
        """

        results = [r for r in results if r is not None]

        if len(results) == 0:
            return None

        if len(results) == 1:
            return results[0]

        stats_out = {}
        for stat in self.stats:
            if stat in ["count", "sum"]:
                stats_out[stat] = np.sum([r[stat] for r in results])
            elif stat == "min":
                stats_out[stat] = np.min([r[stat] for r in results])
            elif stat == "max":
                stats_out[stat] = np.max([r[stat] for r in results])
            elif stat == "mean":
                means = [r["mean"] for r in results if r["mean"] is not None]
                counts = [r["count"] for r in results if r["mean"] is not None]
                if np.sum(counts) == 0:
                    stats_out[stat] = None
                else:
                    stats_out[stat] = np.average(means, weights=counts)

        return stats_out

    def _compute_stats(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader, window: rio.windows.Window):
        """Method to be implemented by the subclass to compute the statistics of a single geometry.
        """
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

    def __call__(self, raster_file_path: str, vector_file_path: str, stats: list):
        """User-facing method to compute the zonal statistics.

        Parameters
        ----------
        raster_file_path : str OR rio.DatasetReader
        vector_file_path : str OR gpd.GeoDataFrame
        stats : List[str]

        Returns
        -------
        List[Dict[str, float]]
        """
        if "mean" in stats and "count" not in stats:
            stats.append("count")

        self.stats = stats
        self.raster_file_path = raster_file_path
        self.vector_file_path = vector_file_path
        
        if isinstance(vector_file_path, str):
            vector_layer = gpd.read_file(self.vector_file_path)
        elif isinstance(vector_file_path, gpd.GeoDataFrame):
            vector_layer = vector_file_path
        else:
            raise ValueError("vector_file_path must be a string or a GeoDataFrame")

        # assuming no specific affine, nodata, transform, band=1
        if isinstance(raster_file_path, str):
            with rio.open(self.raster_file_path) as raster:
                results = self._run(vector_layer, raster)
        elif isinstance(raster_file_path, rio.DatasetReader):
            results = self._run(vector_layer, raster_file_path)
        else:
            raise ValueError("raster_file_path must be a string or a DatasetReader")
            
        return results