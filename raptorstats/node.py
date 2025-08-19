from rasterio import windows as rio_windows
from shapely import Geometry


class Node:
    def __init__(self, _id, parent_id, level, _box, transform, stats, max_depth):
        self.id = _id
        self.parent_id = parent_id
        self.level = level
        self.box = _box
        sq = (
            rio_windows.from_bounds(*self.box.bounds, transform=transform)
            .round_offsets()
            .round_lengths()
        )
        r0 = int(sq.row_off)  # top row (inclusive)
        r1 = r0 + int(sq.height)  # bottom row (exclusive)
        c0 = int(sq.col_off)  # left col (inclusive)
        c1 = c0 + int(sq.width)  # right col (exclusive)
        self.pixel_bounds = (r0, r1, c0, c1)
        self.stats = stats
        self.max_depth = max_depth

    def is_contained_in_geom(self, geom: Geometry) -> bool:
        return self.box.within(geom)

    def is_leaf(self) -> bool:
        return self.level == self.max_depth

    def is_root(self) -> bool:
        return self.level == 0
