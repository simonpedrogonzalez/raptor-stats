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

Combining partials
```
Total time: 46.3969 s
219     79702   28198373.0    353.8     60.8                      partials_per_feature[f_index].append(self.stats.from_array(pixel_values))
231       343    4459274.9  13000.8      9.6              r = self.stats.from_partials(partials_per_feature[i])
```

Accumulating

```
Total time: 17.145 s
```

Conclusion: do not use partials, obviously.

### numpy split instead of manually slicing for each reading_line

When reading the data from the raster, instead of slicing the row of pixels manually, use `np.split`. No, it can't be used because the reading line might have gaps.