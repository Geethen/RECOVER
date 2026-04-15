# Benchmark Results

Comparison of Earth Engine data extraction methods using 4 parallel workers on a simple raster.

| Method          |   Pixels |   Wall Time (s) |
|:----------------|---------:|----------------:|
| computeFeatures |  1611472 |       423.762   |
| computePixels   |        0 |         2.92808 |
| xee             |        0 |         4.57166 |
