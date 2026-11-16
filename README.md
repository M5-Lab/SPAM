# SPAM — Spectral Alignment of Mesogens

SPAM resolves the head–tail ambiguity of a nematic director field through a global spectral relaxation. This repository contains the current standalone implementation and a reproducible [OVITO Nematics](https://github.com/Sapta98/Nematics) pipeline.

## Method

For particles within a cutoff radius, SPAM constructs the sparse interaction matrix

$$A_{ij}=w_{ij}(\mathbf n_i\cdot\mathbf n_j),$$

computes its largest-magnitude eigenvector, and uses its component signs to orient the directors. Gaussian and cubic-spline radial weights are supported.

## Standalone quick start

```bash
python -m pip install -r requirements.txt
python SPAM.py examples/directors.txt --cutoff 4.5 --output output/spam.txt
```

Input is a whitespace-delimited `x y z nx ny nz` table. Output adds the SPAM `flag`.

## Full OVITO pipeline

```bash
ovitos -m pip install --user git+https://github.com/Sapta98/Nematics.git
ovitos pipelines/run_spam.py examples/Plushalf output/plushalf_spam.dump --cutoff 4.5
```

This reads the LAMMPS dump, converts quaternion orientations, runs SPAM, constructs the discontinuity layer, and runs DLA. Pass `--no-dla` to stop before defect reconstruction.

## Representative examples

- `examples/Plushalf` — planar +1/2 defect.
- `examples/Comet` — three-dimensional comet configuration.
- `examples/directors.txt` — compact standalone smoke-test input.

The LAMMPS examples come from Nematics. For the complete GUI workflow and all defect examples, see [OVITO Nematics](https://github.com/Sapta98/Nematics).

## License

MIT. See [LICENSE](LICENSE).
