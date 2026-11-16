"""Run the current SPAM -> discontinuity layer -> DLA OVITO pipeline."""
import argparse
from ovito.io import export_file, import_file
from ovito_nematics import NematicDefectAnalysisModifier

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--cutoff", type=float, default=4.5)
parser.add_argument("--weight-kernel", choices=("Gaussian", "Cubic spline"), default="Gaussian")
parser.add_argument("--no-dla", action="store_true")
args = parser.parse_args()
pipeline = import_file(args.input)
modifier = NematicDefectAnalysisModifier()
modifier.analysis_method = "SPAM"
modifier.orientation_source = "Raw LAMMPS quaternion properties"
modifier.neighborhood_cutoff = args.cutoff
modifier.weight_kernel = args.weight_kernel
modifier.write_discontinuity_layer = True
modifier.identify_disclinations = not args.no_dla
pipeline.modifiers.append(modifier)
export_file(pipeline, args.output, "lammps/dump", columns=[
    "Particle Identifier", "Position.X", "Position.Y", "Position.Z",
    "Nematic Director.X", "Nematic Director.Y", "Nematic Director.Z",
    "Directors.X", "Directors.Y", "Directors.Z", "Flag", "Boundary",
])
print(pipeline.compute().attributes.get("Nematics.Status", "SPAM pipeline complete"))
