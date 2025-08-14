# Profiling notes

## Scanline

### np.asarray vs shapely.get_parts

Inconsistent resutls, probably mostly the same.

```
Total time: 6.85498 s
   117       245     761042.1   3106.3     11.1              g_arr = shapely.get_parts(geoms)
```

```
Total time: 7.0705 s
   116       245     815810.0   3329.8     11.5              g_arr = np.asarray(geoms, dtype=object)
```

### Prepare geometries

Inconsistent results, probably mostly the same

```
   103     22309     158725.7      7.1      0.2          for g in features.geometry:
   104     22302     117651.7      5.3      0.2              shapely.prepare(g)
   105         7     572145.9  81735.1      0.7          scanlines = MultiLineString(list(all_points))
   106         7        701.7    100.2      0.0          shapely.prepare(scanlines)
   107                                           
   108         7    8644796.8 1.23e+06     11.2          all_intersections = scanlines.intersection(features.geometry)
```
```
   105         7     575124.2  82160.6      0.7          scanlines = MultiLineString(list(all_points))
   107                                           
   108         7    9259383.3 1.32e+06     11.7          all_intersections = scanlines.intersection(features.geometry)
```

### Excessive np.ma.concatenate calls ?

Idk if there's a better way to do this because I still need to build an np.ma.array from a list of arrays.

```
   223       245     357354.8   1458.6      5.0                  feature_data = np.ma.concatenate(pixel_values_per_feature[i])
```

### Combining partials instead of accumulating

Calculating partials at each line reading and combine them, instead of extracting the pixel values and calculating the statistics from the accumulated pixel values at the end. Turns out that calculating partials at each line is much slower (but possibly more memory efficient). The conclusion is that accumulating will be the behavior for now.

```
Total time: 46.3969 s
219     79702   28198373.0    353.8     60.8                      partials_per_feature[f_index].append(self.stats.from_array(pixel_values))
231       343    4459274.9  13000.8      9.6              r = self.stats.from_partials(partials_per_feature[i])
```

Accumulating

```
Total time: 17.145 s
```


### numpy split instead of manually slicing for each reading_line

When reading the data from the raster, instead of slicing the row of pixels manually, use `np.split`. No, it can't be used because the reading line might have gaps.

## AggQuadTree

### TL;DR

The bottleneck of the agqt method are the scanline functions, so the agqt method isn't adding any significant overhead (apart from building the tree, which should be done once per raster, and it's bottleneck are also the scanline functions).

### Building the tree takes the most amount of time

Luckily we have to only do it once per raster. `compute_quad_tree` can be improved a little bit, but the bottleneck is the 1st level statistics computation, which is done only once, and it's just running scanline on the whole raster using a grid of squares (nodes) as `vector_layer`. So there's not much to optimize here.

```
   # Precomputations takes up 88% of the time
   261                                               def _run(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
   262         7  140813255.5 2.01e+07     88.5          self._precomputations(features, raster)
   
   # Compute quad tree takes 94% of that
   248                                               def _precomputations(self, features: gpd.GeoDataFrame, raster: rio.DatasetReader):
   249         7         73.6     10.5      0.0          if self._should_build_indices():
   250         7       1499.5    214.2      0.0              if self._index_files_exist():
   251        14       2194.8    156.8      0.0                  warnings.warn(
   252         7         10.3      1.5      0.0                      f"Index files {self.index_path}.idx and {self.index_path}.dat already exist. They will be overwritten.",
   253                                                           )
   254         7       3588.0    512.6      0.0                  self._delete_index_files()
   255         7  133657465.1 1.91e+07     94.9              self._compute_quad_tree(features, raster)
   256                                                   else:
   257                                                       self._load_quad_tree()
   258         7    7148153.2 1.02e+06      5.1          self._compute_scanline_reading_table(features, raster)

   # Running scanline takes 89% of that
   199         7  119853461.6 1.71e+07     89.8                  aggregates = np.array(sc(raster, vector_layer, self.stats))
```

### When the tree is already built

When using the index file, the precomputations only consist of loading the tree and computing the scanline reading table:

```
   # Precomputations takes 25% of the time
   262         7    5040792.6 720113.2     24.7          self._precomputations(features, raster)

   # Compute reading table is all of that
   257         7       7948.6   1135.5      0.2              self._load_quad_tree()
   258         7    5030856.5 718693.8     99.8          self._compute_scanline_reading_table(features, raster)
```

When runnning the method, the lines that take the bulk of the time are:

```
   # I can't really improve rtree intersection performance, and also it's only 8%
   294       343    1578053.4   4600.7      7.7              nodes = list(self.idx.intersection(window_bounds, objects=True))[::-1]

   # This is the shapely box.within(geom) check, takes 1.2% of the time, I don't think I can improve it
   296       343     245591.6    716.0      1.2              nodes = [node for node in nodes if node.is_contained_in_geom(geom)]

   # This is running scanline, so nothing to improve here regarding aggqt
   509        14   13352140.1 953724.3     65.5          self.results = process_reading_table(
   510         7         51.8      7.4      0.0              reading_table, features, raster, self.stats, partials=partials
   511                                                   )
```

