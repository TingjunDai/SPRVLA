import argparse
import logging
import time

from olmo.data.robot_datasets import *
from olmo.data.lvis_dataset import LVIS

from olmo.util import prepare_cli_environment

PRETRAIN_DATASETS = [
    BC_Z, BridgeDataV2, RT_1, AuxiliaryDepthData, AuxiliaryTraceData, LVIS
]

MIDTRAIN_DATASETS = [
    SPRVLADatasetHomePrimary, SPRVLADatasetHomeSecondary, SPRVLADatasetTabletopPrimary, SPRVLADatasetTabletopSecondary
]

LIBERO_DATASETS = [
    LIBEROSpatial, LIBEROObject, LIBEROGoal, LIBEROLong
]

DATASETS = PRETRAIN_DATASETS + MIDTRAIN_DATASETS + LIBERO_DATASETS


DATASET_MAP = {
    x.__name__.lower().replace("_", ""): x for x in DATASETS
}


def download():
    parser = argparse.ArgumentParser(prog="Download SPRVLA datasets")
    parser.add_argument("dataset",
                        help="Datasets to download, can be a name or one of: all, pixmo, academic or academic_eval")
    parser.add_argument("--n_procs", type=int, default=1,
                        help="Number of processes to download with")
    parser.add_argument("--ignore_errors", action="store_true",
                        help="If dataset fails to download, skip it and continue with the remaining")
    args = parser.parse_args()

    prepare_cli_environment()

    if args.dataset == "all":
        to_download = DATASETS
    elif args.dataset == "pretrain":
        to_download = PRETRAIN_DATASETS
    elif args.dataset == "midtrain":
        to_download = MIDTRAIN_DATASETS
    elif args.dataset == "libero":
        to_download = LIBERO_DATASETS
    elif args.dataset.lower().replace("_", "") in DATASET_MAP:
        to_download = [DATASET_MAP[args.dataset.lower().replace("_", "")]]
    else:
        raise NotImplementedError(args.dataset)

    for ix, dataset in enumerate(to_download):
        t0 = time.perf_counter()
        logging.info(f"Starting download for {dataset.__name__} ({ix+1}/{len(to_download)})")
        try:
            dataset.download(n_procs=args.n_procs)
        except Exception as e:
            if args.ignore_errors:
                logging.warning(f"Error downloading {dataset.__name__}: {e}")
                continue
            else:
                raise e
        logging.info(f"Done with {dataset.__name__} in {time.perf_counter()-t0:0.1f} seconds")


if __name__ == '__main__':
    download()
