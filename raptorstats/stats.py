import collections
import numpy as np

DEFAULT_STATS = ["count", "min", "max", "mean"]

VALID_STATS = DEFAULT_STATS + [
    "sum",
    "std",
    "median",
    "majority",
    "minority",
    "unique",
    "range",
    "nodata",
    "nan",
]

class Stats:
    """Configuration for statistics computation.

    Stores the statistics to be computed.
    """

    def __init__(self, stats=None, categorical=False):
        if not stats:
            stats = [] if categorical else DEFAULT_STATS
        elif isinstance(stats, str):
            if stats in ["*", "ALL"]:
                stats = VALID_STATS
            else:
                stats = stats.split()
        
        percentiles = []
        for token in stats:
            if token.startswith("percentile_"):
                percentiles.append(self._get_percentile(token))
            elif token not in VALID_STATS:
                raise ValueError(
                    "Stat `%s` not valid; " "must be one of \n %r" % (token, VALID_STATS)
                )

        requested = set(stats)
        required = set(stats)
        required.update({ "count" }) # count is always useful

        if {"mean", "std", "median"} & required or percentiles:
            required.update({"count", "sum"})
        if "std" in required:
            required.update({"sum_sq"})
        if "range" in required:
            required.update({"min", "max"})
        
        run_count = categorical or \
            percentiles or \
            required & {"majority", "minority", "unique", "median"}

        ordered = []
        for tok in (stats + VALID_STATS):
            if tok in required and tok not in ordered:
                ordered.append(tok)
                required.remove(tok)
        for tok in required:
            ordered.append(tok)

        self.requested = requested
        self.stats = ordered
        self.categorical = categorical
        self.run_count = run_count
        self.percentiles = percentiles

    def clean_results(self, results):
        
        for res in results:
            if not res:
                continue
            for k in list(res.keys()):
                if self.categorical and k == "histogram":
                    continue
                if k not in self.requested:
                    del res[k]
        return results

    def _get_percentile(self, stat):
        if not stat.startswith("percentile_"):
            raise ValueError("must start with 'percentile_'")
        qstr = stat.replace("percentile_", "")
        q = float(qstr)
        if q > 100.0:
            raise ValueError("percentiles must be <= 100")
        if q < 0.0:
            raise ValueError("percentiles must be >= 0")
        return q

    def __repr__(self):
        return f"StatsConfig(stats={self.stats}, categorical={self.categorical}, run_count={self.run_count})"

    def from_array(self, data: np.ma.MaskedArray) -> dict:
        """
        Compute the configured statistics on *data*.

        Parameters
        ----------
        data : np.ndarray or np.ma.MaskedArray
            Pixel values for one feature.  Masked or NaN values are treated
            as nodata.

        Returns
        -------
        dict
            Keys are the statistic names requested at construction time
            (plus every {value: count} entry if categorical=True).
            Values are floats or ints; empty inputs yield None.
        """
        # ---------- normalise to plain ndarray + boolean mask -----------
        if np.ma.isMaskedArray(data):
            mask = data.mask
            arr  = data.data
        else:
            arr  = np.asarray(data)
            mask = np.zeros(arr.shape, dtype=bool)

        nan_mask   = np.isnan(arr)
        nodata_mask = mask | nan_mask
        valid      = arr[~nodata_mask]

        out = { s : np.nan for s in self.stats }

        if "nodata" in self.stats:
            out["nodata"] = int(mask.sum())
        if "nan" in self.stats:
            out["nan"] = int(nan_mask.sum())

        n = valid.size
        if "count" in self.stats:
            out["count"] = int(n)

        if n:
            # Basic stats
            if "sum" in self.stats:
                out["sum"] = float(valid.sum())
            if "min" in self.stats:
                out["min"] = float(valid.min())
            if "max" in self.stats:
                out["max"] = float(valid.max())
            if "mean" in self.stats:
                out["mean"] = float(valid.mean())
            if "range" in self.stats:
                out["range"] = float(valid.max() - valid.min())

            sum_sq = None
            if "sum_sq" in self.stats:
                sum_sq = (valid ** 2).sum()
                out["sum_sq"] = float(sum_sq)
            if "std" in self.stats:
                out["std"] = float(
                    np.sqrt((sum_sq - (valid.sum() ** 2) / n) / n)
                )
            if "median" in self.stats:
                out["median"] = float(np.median(valid))

            # Percentiles
            for tok in self.percentiles:
                out["percentile_" + str(int(tok))] = float(np.percentile(valid, tok))

            # Histogram / Categorical stats
            if self.run_count:
                vals, counts = np.unique(valid, return_counts=True)
                counter = dict(zip(vals.tolist(), counts.tolist()))

                if "unique" in self.stats:
                    out["unique"] = len(counter)

                if "majority" in self.stats and counter:
                    out["majority"] = float(max(counter, key=counter.get))
                if "minority" in self.stats and counter:
                    out["minority"] = float(min(counter, key=counter.get))

                out["histogram"] = counter

        elif self.run_count:
            out["histogram"] = {}

        return out
    
      # ------------------------------------------------------------------ #

    def from_partials(self, partials: list[dict]) -> dict:
        """
        Combine already-computed chunk statistics into one feature-level dict.

        Each element of *partials* must have been produced by `from_array()`
        with **the same Stats configuration**.

        Returns
        -------
        dict
            Feature-level statistics – exactly the same schema you get from
            calling `from_array()` on the full dataset at once.
        """
        # discard empty / None chunks
        chunks = [p for p in partials if p]

        # nothing at all
        if not chunks:
            out = {stat: np.nan for stat in self.stats}
            out["count"] = 0
            return out

        if len(chunks) == 1:
            return chunks[0].copy()

        # helpers to accumulate
        total_count   = 0
        total_sum     = 0.0
        total_sum_sq  = 0.0
        total_nodata  = 0
        total_nan     = 0
        global_min    = np.inf
        global_max    = -np.inf
        histogram     = collections.Counter()

        for r in chunks:
            c = r.get("count", 0) or 0
            s = r.get("sum", 0.0) or 0.0
            ss= r.get("sum_sq", 0.0) or (r.get("std") and c and
                                        (r["std"]**2 + (s/c)**2) * c) or 0.0

            total_count  += c
            total_sum    += s
            total_sum_sq += ss
            total_nodata += r.get("nodata", 0) or 0
            total_nan    += r.get("nan", 0) or 0
            if "min" in r and not np.isnan(r["min"]):
                global_min = min(global_min, r["min"])
            if "max" in r and not np.isnan(r["max"]):
                global_max = max(global_max, r["max"])
            if "histogram" in r:
                histogram.update(r["histogram"])

        out = {stat: np.nan for stat in self.stats}

        if "nodata" in self.stats:
                out["nodata"] = int(total_nodata)
        if "nan" in self.stats:
            out["nan"] = int(total_nan)

        if "count" in self.stats:
            out["count"] = int(total_count)

        if total_count:
            if "sum" in self.stats:
                out["sum"] = float(total_sum)
            if "min" in self.stats:
                out["min"] = float(global_min)
            if "max" in self.stats:
                out["max"] = float(global_max)
            if "mean" in self.stats:
                out["mean"] = float(total_sum / total_count)
            if "range" in self.stats:
                out["range"] = float(global_max - global_min)

            if "sum_sq" in self.stats:
                out["sum_sq"] = float(total_sum_sq)
            if "std" in self.stats and total_count:
                var = (total_sum_sq - (total_sum ** 2) / total_count) / total_count
                out["std"] = float(np.sqrt(max(var, 0.0)))  # guard FP jitter

            if "median" in self.stats or self.percentiles:
                # build cumulative distribution
                vals, cnts = zip(*sorted(histogram.items()))
                cum = np.cumsum(cnts)
                def _quantile(q):
                    target = q / 100.0 * total_count
                    idx = np.searchsorted(cum, target, side="left")
                    return float(vals[min(idx, len(vals)-1)])
                if "median" in self.stats:
                    out["median"] = _quantile(50.0)
                for q in self.percentiles:
                    out[f"percentile_{int(q)}"] = _quantile(q)

            if self.run_count:
                if "unique" in self.stats:
                    out["unique"] = len(histogram)
                if "majority" in self.stats:
                    out["majority"] = float(max(histogram, key=histogram.get))
                if "minority" in self.stats:
                    out["minority"] = float(min(histogram, key=histogram.get))
                out['histogram'] = dict(histogram)
        elif self.run_count:
            # empty input, but categorical requested
            out["histogram"] = dict(histogram)

        return out
