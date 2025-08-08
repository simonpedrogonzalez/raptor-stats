from raptorstats.zone_stat_method import ZonalStatMethod
import rasterio as rio
from rasterio.mask import mask as rio_mask
import geopandas as gpd
import shapely as shp
import numpy as np
from rasterstats import zonal_stats


class RasterStatsMasking(ZonalStatMethod):

    __name__ = "RasterStatsMasking"

    def __init__(self):
        super().__init__()

    def _run(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
        return zonal_stats(
            raster,
            features,
            stats=self.stats,
            categorical=True,
            # all_touched=True,
        )

class Masking(ZonalStatMethod):

    __name__ = "Masking"

    def __init__(self):
        super().__init__()

    def mask(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader, window: rio.windows.Window):
        """Create the mask used for the zonal statistics, useful for debugging.
        """
        if isinstance(feature, gpd.GeoDataFrame):
            feature = feature.geometry[0]
        mask = rio.features.rasterize(
            [(shp.geometry.mapping(feature), 1)],
            out_shape=(
                int(np.ceil(window.height)),
                int(np.ceil(window.width)),    
            ),
            transform=rio.windows.transform(window, raster.transform),
            fill=0,
            dtype='uint8'
        ).astype(bool)

        return mask

    def _compute_stats(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader, window: rio.windows.Window):
        """Computes the zonal statistics for a feature using the Masking method.

        In this case, 1 feature, is virtually the same as the clipping method. The difference lays in the
        several features case where each feature has a different value in the mask.
        """
        
        mask = rio.features.rasterize(
            [(shp.geometry.mapping(feature.geometry[0]), 1)],
            out_shape=(window.height, window.width),
            transform=rio.windows.transform(window, raster.transform),
            fill=0,
            dtype='uint8'
        ).astype(bool)

        data = raster.read(window=window, masked=True)
        data = data[:, mask]
        return self._compute_stats_from_masked_array(data)


class Clipping(ZonalStatMethod):
    
    __name__ = "Clipping"

    def __init__(self):
        super().__init__()

    def _compute_stats(self, feature: gpd.GeoDataFrame, raster: rio.DatasetReader, window: rio.windows.Window):

        # nodata_value = raster.nodata if raster.nodata is not None else -999

        data = raster.read(window=window, masked=True)
        clipped_data, _ = rio_mask(raster, [feature.geometry[0]], crop=False, filled=True)
        
        return self._compute_stats_from_array(clipped_data[0]) # Return first band
