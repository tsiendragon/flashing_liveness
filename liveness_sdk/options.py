import argparse

parser = argparse.ArgumentParser(description="liveness sdk options.")
parser.add_argument(
    "--draw_bbox",
    action="store_true",
    help="whether show bbox visualization in UI)",
)
parser.add_argument(
    "--config",
    help="config file",
    default="configs/config.yaml",
)

options = parser.parse_args()
