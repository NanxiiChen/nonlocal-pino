"""Process entry point for field logging, separate from the TensorBoard server."""
import argparse
import runpy
from examples.fracture2d.tensorboard_monitor import main


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();main(runpy.run_path(args.config)['get_config'](),args.config)
