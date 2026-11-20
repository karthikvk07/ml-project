"""Operating-support diagnostics: 2D (speed, brake) convex hulls and 3D ambient k-NN OOD."""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree


class Hull2D:
    def __init__(self, points=None, vertices=None):
        if vertices is None:
            pts = np.asarray(points, float)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) < 3:
                raise ValueError("Need >=3 finite points for a hull")
            vertices = pts[ConvexHull(pts).vertices]
        self.vertices = np.asarray(vertices, float)
        self.tri = Delaunay(self.vertices)

    def contains(self, points):
        p = np.asarray(points, float)
        ok = np.isfinite(p).all(axis=1)
        out = np.zeros(len(p), bool)
        if ok.any():
            out[ok] = self.tri.find_simplex(p[ok]) >= 0
        return out


def union_contains(hulls, points):
    m = np.zeros(len(points), bool)
    for h in hulls:
        m |= h.contains(points)
    return m


class AmbientOOD:
    """Standardized 3D ambient k-NN distance vs reference leave-one-out quantile."""

    def __init__(self, ref_ambient, k=5, q=0.99):
        A = np.asarray(ref_ambient, float)
        A = A[np.isfinite(A).all(axis=1)]
        self.mean, self.std, self.k = A.mean(axis=0), A.std(axis=0), k
        self.ref = (A - self.mean) / self.std
        self.tree = cKDTree(self.ref)
        loo = self.tree.query(self.ref, k=k + 1)[0][:, -1]
        self.threshold = float(np.quantile(loo, q))
        self.ref_median = float(np.median(loo))

    def distance(self, ambient):
        a = np.asarray(ambient, float)
        out = np.full(len(a), np.nan)
        ok = np.isfinite(a).all(axis=1)
        if ok.any():
            out[ok] = self.tree.query((a[ok] - self.mean) / self.std, k=self.k)[0][:, -1]
        return out

    def flag(self, ambient):
        d = self.distance(ambient)
        return d, np.where(np.isfinite(d), d > self.threshold, True)
