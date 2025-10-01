# SPAM - Spatial Pattern Analysis Method

A Python implementation of the SPAM (Spatial Pattern Analysis Method) for analyzing nematic liquid crystal systems using Gaussian weights and spectral analysis.

## Overview

SPAM is a computational method for analyzing spatial patterns in nematic liquid crystal systems. This implementation uses Gaussian weights for neighbor interactions and performs spectral analysis on the resulting adjacency matrix to identify dominant spatial patterns.

## Features

- **Class-based Architecture**: Clean, modular design with the `SPAMAnalyzer` class
- **Gaussian Weights**: Uses Gaussian weight functions with automatic parameter calculation
- **Automatic Parameter Calculation**: 
  - Search radius = 1.5 × long_axis_length
  - Gaussian sigma = search radius
- **Comprehensive Output**: Saves results, eigenvalues, and eigenvectors separately
- **CLI Interface**: Easy-to-use command-line interface
- **Minimal Dependencies**: Only requires NumPy, SciPy, and standard library

## Installation

### Requirements

```bash
pip install numpy scipy
```

### Dependencies

- `numpy` - Numerical computations
- `scipy.spatial.KDTree` - Efficient neighbor finding
- `scipy.sparse.linalg.eigsh` - Eigenvalue computation
- `scipy.sparse.coo_matrix` - Sparse matrix operations
- Standard library: `sys`, `os`, `argparse`

## Usage

### Command Line Interface

```bash
python3 SPAM.py --k 5 --long-axis 2.0 --file data.txt --output results.txt
```

#### Arguments

- `--k` (required): Number of eigenvalues/eigenvectors to compute
- `--long-axis` (required): Long axis length of uniaxial mesogen
- `--file` (required): Input file containing positions and directors
- `--output` (optional): Output file prefix (default: spam_output.txt)

### Programmatic Usage

```python
from SPAM import SPAMAnalyzer

# Create analyzer
analyzer = SPAMAnalyzer(
    k=5,
    long_axis=2.0,
    file_path="data.txt",
    output_path="results.txt"
)

# Run analysis
result = analyzer.run_analysis()
```

## Input File Format

The input file should contain space-separated values with the following format:

```
x y z nx ny nz
x y z nx ny nz
...
```

Where:
- `x y z`: Particle positions
- `nx ny nz`: Director vectors (normalized orientation vectors)

### Example Input File

```
0.0 0.0 0.0 1.0 0.0 0.0
1.0 0.0 0.0 0.0 1.0 0.0
0.0 1.0 0.0 0.0 0.0 1.0
...
```

## Output Files

The analysis generates three output files:

### 1. Modified Director Field (`output.txt`)
Contains the modified director field combining original positions with SPAM analysis results:
```
# Modified director field (x y z modified_nx modified_ny modified_nz): k=5, long_axis=2.0, radius=3.0, sigma=3.0
x_1 y_1 z_1 modified_nx_1 modified_ny_1 modified_nz_1
x_2 y_2 z_2 modified_nx_2 modified_ny_2 modified_nz_2
x_3 y_3 z_3 modified_nx_3 modified_ny_3 modified_nz_3
...
```

This file has the same format as the input file but with the director vectors replaced by the SPAM analysis results.

### 2. Eigenvalues File (`output_eigenvalues.txt`)
Contains the computed eigenvalues:
```
# SPAM eigenvalues: k=5, long_axis=2.0, radius=3.0, sigma=3.0
eigenvalue_1
eigenvalue_2
eigenvalue_3
...
```

### 3. Eigenvectors File (`output_eigenvectors.txt`)
Contains the raw numeric eigenvectors:
```
# SPAM eigenvectors: k=5, long_axis=2.0, radius=3.0, sigma=3.0
eigenvec_1_comp_1 eigenvec_1_comp_2 ... eigenvec_1_comp_n
eigenvec_2_comp_1 eigenvec_2_comp_2 ... eigenvec_2_comp_n
...
```

## Algorithm Details

### Parameter Relationships

- **Long Axis Length**: User-specified input parameter
- **Search Radius**: Automatically calculated as `1.5 × long_axis_length`
- **Gaussian Sigma**: Set equal to the search radius for optimal weighting

### Workflow

1. **Data Loading**: Read positions and directors from input file
2. **Neighbor Finding**: Use KDTree to find neighbors within search radius
3. **Weight Calculation**: Compute Gaussian weights based on distances
4. **Matrix Construction**: Build weighted adjacency matrix from dot products
5. **Spectral Analysis**: Compute largest eigenvalues and eigenvectors
6. **Result Calculation**: Apply sign function and compute final SPAM result
7. **Output**: Save results, eigenvalues, and eigenvectors to files

### Mathematical Foundation

The SPAM method constructs a weighted adjacency matrix A where:
- A_ij = w_ij × (n_i · n_j)
- w_ij = exp(-d_ij² / (2σ²)) (Gaussian weights)
- d_ij = distance between particles i and j
- n_i, n_j = director vectors
- σ = search radius

The method then computes the largest eigenvalues and eigenvectors of this matrix to identify dominant spatial patterns.

## Class Reference

### SPAMAnalyzer

#### Constructor
```python
SPAMAnalyzer(k, long_axis, file_path, output_path='spam_output.txt')
```

#### Methods

- `load_data()`: Load positions and directors from file
- `find_neighbors()`: Find neighbors within search radius
- `calculate_weights()`: Compute Gaussian weights
- `build_adjacency_matrix()`: Build weighted adjacency matrix
- `compute_eigenvalues()`: Compute eigenvalues and eigenvectors
- `compute_result()`: Calculate final SPAM result
- `save_results()`: Save all outputs to files
- `run_analysis()`: Execute complete analysis workflow

#### Attributes

- `k`: Number of eigenvalues/eigenvectors
- `long_axis`: Long axis length of mesogen
- `radius`: Search radius (1.5 × long_axis)
- `sigma`: Gaussian standard deviation (equal to radius)
- `positions`: Particle positions
- `directors`: Director vectors
- `eigenvalues`: Computed eigenvalues
- `eigenvectors`: Raw eigenvectors
- `result`: Final SPAM result

## Examples

### Basic Analysis

```bash
python3 SPAM.py --k 10 --long-axis 1.5 --file nematic_data.txt
```

This will create:
- `spam_output.txt` - Modified director field
- `spam_output_eigenvalues.txt` - Eigenvalues
- `spam_output_eigenvectors.txt` - Eigenvectors

### Custom Output

```bash
python3 SPAM.py --k 5 --long-axis 2.0 --file data.txt --output my_analysis.txt
```

This will create:
- `my_analysis.txt` - Modified director field
- `my_analysis_eigenvalues.txt` - Eigenvalues  
- `my_analysis_eigenvectors.txt` - Eigenvectors

### Programmatic Example

```python
# Analyze with different parameters
analyzer = SPAMAnalyzer(k=8, long_axis=2.5, file_path="system.txt")
result = analyzer.run_analysis()

# Access intermediate results
print(f"Found {len(analyzer.neighbors)} neighbor pairs")
print(f"Largest eigenvalue: {analyzer.eigenvalues[0]:.3f}")
```

## Performance Considerations

- **Neighbor Finding**: Uses efficient KDTree for O(N log N) neighbor search
- **Sparse Matrices**: Leverages sparse matrix operations for memory efficiency
- **Eigenvalue Computation**: Uses ARPACK for efficient computation of largest eigenvalues
- **Memory Usage**: Scales approximately O(N²) for dense systems, O(N) for sparse systems

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure input file exists and path is correct
2. **Invalid Parameters**: k and long_axis must be positive numbers
3. **Memory Issues**: For large systems, consider reducing k or using more efficient data structures

### Error Messages

- `"k must be a positive integer"`: Set k to a positive integer
- `"long_axis must be a positive number"`: Set long_axis to a positive float
- `"File 'filename' does not exist"`: Check file path and existence

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 SPAM Implementation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## References

SPAM method for analyzing spatial patterns in nematic liquid crystal systems using spectral analysis of weighted adjacency matrices.

