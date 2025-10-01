import numpy as np
from scipy.spatial import KDTree
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
import sys
import os
import argparse


class SPAMAnalyzer:
	"""
	SPAM (Spatial Pattern Analysis Method) analyzer using Gaussian weights.
	
	This class encapsulates all parameters and methods for SPAM analysis,
	with sigma automatically set equal to the search radius.
	"""
	
	def __init__(self, k, long_axis, file_path, output_path='spam_output.txt'):
		"""
		Initialize SPAM analyzer.
		
		Args:
			k (int): Number of eigenvalues/eigenvectors to compute
			long_axis (float): Long axis length of uniaxial mesogen
			file_path (str): Path to input file containing positions and directors
			output_path (str): Path to output file for results
		"""
		self.k = k
		self.long_axis = long_axis
		self.file_path = file_path
		self.output_path = output_path
		
		# Calculate derived parameters
		self.radius = 1.5 * self.long_axis  # Search radius = 1.5 * long_axis
		self.sigma = self.radius  # Standard deviation = radius
		
		# Data storage
		self.positions = None
		self.directors = None
		self.neighbors = None
		self.weights = None
		self.eigenvalues = None
		self.eigenvectors = None
		self.result = None
		
		# Validation
		self._validate_parameters()
		
	def _validate_parameters(self):
		"""Validate input parameters."""
		if self.k <= 0:
			raise ValueError("k must be a positive integer")
		if self.long_axis <= 0:
			raise ValueError("long_axis must be a positive number")
		if not os.path.exists(self.file_path):
			raise FileNotFoundError(f"File '{self.file_path}' does not exist")
	
	def load_data(self):
		"""
		Load positions and directors from file.
		
		Expected file format: x y z nx ny nz (space-separated)
		"""
		print(f"Reading data from {self.file_path}...")
		with open(self.file_path, 'r') as file:
			data = file.readlines()
		
		data = np.array([list(map(float, dat.strip().split())) for dat in data])
		self.positions = data[:, :3]
		self.directors = data[:, 3:]
		print(f"Loaded {len(self.positions)} particles")
		
	def find_neighbors(self):
		"""Find neighbors within the search radius."""
		print(f"Finding neighbors within radius {self.radius}...")
		KD = KDTree(self.positions)
		neighbors = KD.query_ball_point(self.positions, self.radius, workers=-1)
		self.neighbors = np.array([[i, neigh] for i, neighbor in enumerate(neighbors) for neigh in neighbor])
		print(f"Found {len(self.neighbors)} neighbor pairs")
		
	def calculate_weights(self):
		"""
		Calculate Gaussian weights for neighbor pairs.
		
		Uses sigma = radius for the Gaussian weight function.
		"""
		print(f"Calculating Gaussian weights with sigma = {self.sigma}...")
		distances = np.linalg.norm(self.positions[self.neighbors][:, 0] - self.positions[self.neighbors][:, 1], axis=1)
		# Gaussian weight: exp(-distance^2 / (2 * sigma^2))
		self.weights = np.exp(-(distances**2) / (2 * self.sigma**2))
		
	def build_adjacency_matrix(self):
		"""Build weighted adjacency matrix from dot products and weights."""
		print("Building adjacency matrix...")
		rows = self.neighbors[:, 0]
		cols = self.neighbors[:, 1]
		dot_products = np.einsum('ij,ij->i', self.directors[rows], self.directors[cols])
		dot_products *= self.weights
		A_sparse = coo_matrix((dot_products, (rows, cols)), shape=(len(self.positions), len(self.positions)))
		return A_sparse.tocsr()
		
	def compute_eigenvalues(self, A_matrix):
		"""Compute largest eigenvalues and eigenvectors."""
		print(f"Computing {self.k} largest eigenvalues and eigenvectors...")
		self.eigenvalues, eigenvector = eigsh(A_matrix, k=self.k, which='LM')
		# Store raw eigenvectors without sign function
		self.eigenvectors = eigenvector.T
		# Apply sign function for result calculation
		eigenvector_signed = np.sign(self.eigenvectors)[:, :, np.newaxis]
		return self.eigenvalues, eigenvector_signed
		
	def compute_result(self, eigenvector):
		"""Compute final SPAM result."""
		print("Computing SPAM result...")
		self.result = eigenvector * self.directors[np.newaxis, :, :]
		return self.result[0]
		
	def save_results(self):
		"""Save results as modified director field and eigenvalues to output files."""
		print(f"Saving modified director field to {self.output_path}...")
		
		# Combine positions and SPAM result to create modified director field
		modified_director_field = np.column_stack([self.positions, self.result])
		header = f'Modified director field (x y z modified_nx modified_ny modified_nz): k={self.k}, long_axis={self.long_axis}, radius={self.radius}, sigma={self.sigma}'
		np.savetxt(self.output_path, modified_director_field, header=header)
		
		# Save eigenvalues to a separate file
		eigenval_path = self.output_path.replace('.txt', '_eigenvalues.txt')
		print(f"Saving eigenvalues to {eigenval_path}...")
		eigenval_header = f'SPAM eigenvalues: k={self.k}, long_axis={self.long_axis}, radius={self.radius}, sigma={self.sigma}'
		np.savetxt(eigenval_path, self.eigenvalues, header=eigenval_header)
		
		# Save eigenvectors to a separate file
		eigenvec_path = self.output_path.replace('.txt', '_eigenvectors.txt')
		print(f"Saving eigenvectors to {eigenvec_path}...")
		eigenvec_header = f'SPAM eigenvectors: k={self.k}, long_axis={self.long_axis}, radius={self.radius}, sigma={self.sigma}'
		np.savetxt(eigenvec_path, self.eigenvectors, header=eigenvec_header)
		
	def print_parameters(self):
		"""Print all parameters used in the analysis."""
		print("\n" + "="*50)
		print("SPAM ANALYSIS PARAMETERS")
		print("="*50)
		print(f"k (eigenvalues/eigenvectors): {self.k}")
		print(f"Long axis length: {self.long_axis}")
		print(f"Search radius: {self.radius} (1.5 × long_axis)")
		print(f"Gaussian sigma: {self.sigma} (equal to radius)")
		print(f"Input file: {self.file_path}")
		print(f"Output files:")
		print(f"  - Modified director field: {self.output_path}")
		print(f"  - Eigenvalues: {self.output_path.replace('.txt', '_eigenvalues.txt')}")
		print(f"  - Eigenvectors: {self.output_path.replace('.txt', '_eigenvectors.txt')}")
		print(f"Number of particles: {len(self.positions) if self.positions is not None else 'Not loaded'}")
		print("="*50)
		
	def run_analysis(self):
		"""
		Run complete SPAM analysis workflow.
		
		Returns:
			numpy.ndarray: SPAM analysis result
		"""
		try:
			self.print_parameters()
			
			# Step 1: Load data
			self.load_data()
			
			# Step 2: Find neighbors
			self.find_neighbors()
			
			# Step 3: Calculate weights
			self.calculate_weights()
			
			# Step 4: Build adjacency matrix
			A_matrix = self.build_adjacency_matrix()
			
			# Step 5: Compute eigenvalues
			eigenvalue, eigenvector = self.compute_eigenvalues(A_matrix)
			
			# Step 6: Compute result
			result = self.compute_result(eigenvector)
			
			# Step 7: Save results
			self.save_results()
			
			print("Analysis complete!")
			return result
			
		except Exception as e:
			print(f"Error during analysis: {e}")
			raise


def main():
	"""Main function with CLI interface for SPAM analysis."""
	parser = argparse.ArgumentParser(description='SPAM (Spatial Pattern Analysis Method) with Gaussian weights')
	parser.add_argument('--k', type=int, required=True, 
	                   help='Number of eigenvalues/eigenvectors to compute')
	parser.add_argument('--long-axis', type=float, required=True,
	                   help='Long axis length of uniaxial mesogen (radius = 1.5 * long_axis, sigma = radius)')
	parser.add_argument('--file', type=str, required=True,
	                   help='Input file containing positions and directors (format: x y z dx dy dz)')
	parser.add_argument('--output', type=str, default='spam_output.txt',
	                   help='Output file for results (default: spam_output.txt)')
	
	args = parser.parse_args()
	
	try:
		# Create SPAM analyzer
		analyzer = SPAMAnalyzer(
			k=args.k,
			long_axis=args.long_axis,
			file_path=args.file,
			output_path=args.output
		)
		
		# Run analysis
		result = analyzer.run_analysis()
		
	except Exception as e:
		print(f"Error: {e}")
		sys.exit(1)


if __name__ == "__main__":
	main()