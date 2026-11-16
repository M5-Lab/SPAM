"""Standalone SPAM sign assignment for nematic director fields."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import KDTree

def normalize_directors(directors):
    directors = np.asarray(directors, dtype=float)
    if directors.ndim != 2 or directors.shape[1] != 3:
        raise ValueError("directors must have shape (N, 3)")
    norms = np.linalg.norm(directors, axis=1)
    if np.any(~np.isfinite(directors)) or np.any(norms <= np.finfo(float).eps):
        raise ValueError("directors must be finite and non-zero")
    return directors / norms[:, None]

def compute_weights(distances, cutoff, kernel):
    if kernel == "gaussian":
        return np.exp(-(distances**2) / (2.0 * cutoff**2))
    if kernel == "cubic-spline":
        x = 2.0 * distances / cutoff
        result = np.zeros_like(x)
        inner, outer = (x >= 0) & (x < 1), (x >= 1) & (x <= 2)
        result[inner] = (3*x[inner]**3 - 6*x[inner]**2 + 4) / 6
        result[outer] = (2-x[outer])**3 / 6
        return result
    raise ValueError(f"unsupported weight kernel: {kernel}")

def run_spam(positions, directors, cutoff, weight_kernel="gaussian"):
    """Return signed directors, +/- flags, and the leading eigenvalue."""
    positions, directors = np.asarray(positions, float), normalize_directors(directors)
    if positions.shape != directors.shape:
        raise ValueError("positions and directors must both have shape (N, 3)")
    if cutoff <= 0:
        raise ValueError("cutoff must be greater than zero")
    if len(positions) <= 1:
        return directors.copy(), np.ones(len(positions)), 0.0
    neighborhoods = KDTree(positions).query_ball_point(positions, cutoff, workers=-1)
    pairs = np.asarray([(i, j) for i, ns in enumerate(neighborhoods) for j in ns], dtype=np.int64).reshape(-1, 2)
    rows, cols = pairs.T
    distances = np.linalg.norm(positions[rows] - positions[cols], axis=1)
    interactions = compute_weights(distances, cutoff, weight_kernel) * np.einsum("ij,ij->i", directors[rows], directors[cols])
    matrix = coo_matrix((interactions, (rows, cols)), shape=(len(positions),)*2).tocsr()
    eigenvalues, eigenvectors = eigsh(matrix, k=1, which="LM")
    flags = np.sign(eigenvectors[:, 0])
    flags[flags == 0] = 1
    return flags[:, None] * directors, flags, float(eigenvalues[0])

def main():
    parser = argparse.ArgumentParser(description="Apply SPAM to an x y z nx ny nz table")
    parser.add_argument("input", type=Path)
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--weight-kernel", choices=("gaussian", "cubic-spline"), default="gaussian")
    parser.add_argument("--output", type=Path, default=Path("spam_output.txt"))
    args = parser.parse_args()
    data = np.loadtxt(args.input)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError("input must contain x y z nx ny nz columns")
    signed, flags, eigenvalue = run_spam(data[:, :3], data[:, 3:], args.cutoff, args.weight_kernel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, np.column_stack((data[:, :3], signed, flags)), header=f"x y z nx ny nz flag | leading_eigenvalue={eigenvalue:.16g}")
    print(f"Wrote {args.output} ({len(data)} particles; eigenvalue={eigenvalue:.8g})")

if __name__ == "__main__":
    main()
