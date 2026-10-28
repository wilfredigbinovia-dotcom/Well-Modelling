"""Well geometry: deviation survey, tubing, temperature profile and the
computational grid used by the pressure traverse."""
from __future__ import annotations

import numpy as np


class Wellbore:
    """Production string from wellhead (MD = 0) to mid-perforations.

    Parameters
    ----------
    survey_md, survey_tvd : deviation survey (ft). Must start at 0.
    tubing_id : flow-path inner diameter (in). Either a single value or a list
        of (md_bottom, id_in) tuples for a tapered string / casing below the
        tubing shoe, ordered from the top.
    perf_md : measured depth of mid-perforations (ft)
    gauge_md : measured depth of the downhole gauge (ft), optional
    roughness : absolute pipe roughness (in)
    wht, bht : flowing wellhead and reservoir temperatures (degF); a linear
        profile in TVD is assumed (screening assumption).
    step : maximum grid spacing along MD (ft)
    """

    def __init__(self, survey_md, survey_tvd, tubing_id, perf_md: float,
                 gauge_md: float | None = None, roughness: float = 0.0006,
                 wht: float = 120.0, bht: float = 200.0, step: float = 300.0):
        self.survey_md = np.asarray(survey_md, float)
        self.survey_tvd = np.asarray(survey_tvd, float)
        if self.survey_md[0] != 0:
            raise ValueError("survey must start at MD = 0")
        if np.any(np.diff(self.survey_md) <= 0):
            raise ValueError("survey MD must be strictly increasing")
        if np.any(np.diff(self.survey_tvd) > np.diff(self.survey_md) + 1e-6):
            raise ValueError("TVD increment cannot exceed MD increment")
        self.perf_md = float(perf_md)
        self.gauge_md = None if gauge_md is None else float(gauge_md)
        self.roughness = float(roughness)
        self.wht = float(wht)
        self.bht = float(bht)
        self.step = float(step)
        if np.isscalar(tubing_id):
            self.id_table = [(self.perf_md, float(tubing_id))]
        else:
            self.id_table = [(float(m), float(d)) for m, d in tubing_id]
        self._build_grid()

    # ------------------------------------------------------------------
    def tvd(self, md):
        """True vertical depth (ft) at measured depth `md`."""
        return self._tvd_extrap(md)

    def _tvd_extrap(self, md):
        md = np.asarray(md, float)
        tvd = np.interp(md, self.survey_md, self.survey_tvd)
        beyond = md > self.survey_md[-1]
        if np.any(beyond):
            dm = self.survey_md[-1] - self.survey_md[-2]
            slope = (self.survey_tvd[-1] - self.survey_tvd[-2]) / dm
            tvd = np.where(beyond, self.survey_tvd[-1] + slope * (md - self.survey_md[-1]), tvd)
        return tvd

    def diameter_in(self, md):
        md = np.asarray(md, float)
        out = np.full(md.shape, self.id_table[-1][1])
        for bottom, d in reversed(self.id_table):
            out = np.where(md <= bottom + 1e-9, d, out)
        return out

    @property
    def perf_tvd(self) -> float:
        return float(self._tvd_extrap(self.perf_md))

    def temperature(self, md):
        tvd = self._tvd_extrap(md)
        return self.wht + (self.bht - self.wht) * tvd / self.perf_tvd

    def _build_grid(self):
        pts = [0.0, self.perf_md]
        pts += [m for m in self.survey_md if 0 < m < self.perf_md]
        pts += [m for m, _ in self.id_table if 0 < m < self.perf_md]
        if self.gauge_md is not None:
            if not 0 < self.gauge_md <= self.perf_md:
                raise ValueError("gauge must lie between surface and perforations")
            pts.append(self.gauge_md)
        pts = np.unique(np.round(pts, 6))
        nodes = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            n = int(np.ceil((b - a) / self.step))
            nodes.extend(np.linspace(a, b, n + 1)[1:])
        self.md = np.array(nodes)
        tvd = self._tvd_extrap(self.md)
        dmd = np.diff(self.md)
        self.dmd = dmd
        self.sin_theta = np.clip(np.diff(tvd) / dmd, -1.0, 1.0)
        mids = 0.5 * (self.md[1:] + self.md[:-1])
        self.seg_d_ft = self.diameter_in(mids) / 12.0
        self.temp = self.temperature(self.md)
        self.gauge_index = (None if self.gauge_md is None
                            else int(np.argmin(np.abs(self.md - self.gauge_md))))

    def index_of(self, md: float) -> int:
        """Grid node nearest to `md`."""
        return int(np.argmin(np.abs(self.md - md)))

    def summary(self) -> dict:
        return {"perf_MD": self.perf_md, "perf_TVD": round(self.perf_tvd, 1),
                "gauge_MD": self.gauge_md,
                "gauge_TVD": None if self.gauge_md is None else round(float(self._tvd_extrap(self.gauge_md)), 1),
                "tubing_ID_in": [d for _, d in self.id_table],
                "max_inclination_deg": round(float(np.degrees(np.arccos(self.sin_theta.min()))), 1),
                "grid_nodes": len(self.md)}
