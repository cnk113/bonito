"""
Bonito Basecaller
"""

import sys
import torch
import numpy as np
from tqdm import tqdm
from time import perf_counter
from datetime import timedelta
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.aligner import Aligner
from bonito.io import CTCWriter, Writer
from bonito.basecall import basecall, ctc_data
from bonito.multiprocessing import process_iter
from bonito.util import column_to_set, get_reads, load_model


def main(args):

    if args.save_ctc and not args.reference:
        sys.stderr.write("> a reference is needed to output ctc training data\n")
        exit(1)

    sys.stderr.write("> loading model\n")
    model = load_model(
        args.model_directory, args.device, weights=int(args.weights),
        chunksize=args.chunksize, use_rt=args.cudart,
    )

    if args.reference:
        sys.stderr.write("> loading reference\n")
        aligner = Aligner(args.reference, preset='ont-map')
        if not aligner:
            sys.stderr.write("> failed to load/build index\n")
            exit(1)
    else:
        aligner = None

    reads = process_iter(
        get_reads(args.reads_directory, read_ids=column_to_set(args.read_ids), skip=args.skip)
    )

    if args.save_ctc:
        data = ctc_data(
            model, reads, aligner,
            min_accuracy=args.ctc_min_accuracy, min_coverage=args.ctc_min_coverage
        )
        writer = CTCWriter(
            tqdm(data, desc="> calling", unit=" reads", leave=False), aligner
        )
    else:
        basecalls = basecall(
            model, reads, aligner=aligner,
            beamsize=1 if args.fastq else args.beamsize,
            chunksize=args.chunksize, overlap=args.overlap,
            batchsize=args.batchsize
        )
        writer = Writer(
            tqdm(basecalls, desc="> calling", unit=" reads", leave=False), aligner, fastq=args.fastq
        )

    t0 = perf_counter()
    writer.start()
    writer.join()
    duration = perf_counter() - t0
    num_samples = sum(num_samples for read_id, num_samples in writer.log)

    sys.stderr.write("> completed reads: %s\n" % len(writer.log))
    sys.stderr.write("> duration: %s\n" % timedelta(seconds=np.round(duration)))
    sys.stderr.write("> samples per second %.1E\n" % (num_samples / duration))
    sys.stderr.write("> done\n")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("reads_directory")
    parser.add_argument("--reference")
    parser.add_argument("--read-ids")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--beamsize", default=5, type=int)
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--chunksize", default=0, type=int)
    parser.add_argument("--overlap", default=0, type=int)
    parser.add_argument("--skip", action="store_true", default=False)
    parser.add_argument("--fastq", action="store_true", default=False)
    parser.add_argument("--cudart", action="store_true", default=False)
    parser.add_argument("--save-ctc", action="store_true", default=False)
    parser.add_argument("--ctc-min-coverage", default=0.9, type=float)
    parser.add_argument("--ctc-min-accuracy", default=0.9, type=float)
    return parser
