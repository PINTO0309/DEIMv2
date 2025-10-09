import argparse
import coremltools as ct

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to SavedModel directory")
parser.add_argument("--sizes", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=[640, 640])
parser.add_argument("--output", required=True, help="Output mlmodel/mlpackage path")
args = parser.parse_args()

height, width = args.sizes
mlmodel = ct.convert(
    args.input,
    source="tensorflow",
    convert_to="mlprogram",
    inputs=[ct.ImageType(name="input_bgr", shape=(1, height, width, 3))],
)
mlmodel.save(args.output)
