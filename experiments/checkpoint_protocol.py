"""Shared checkpoint epoch semantics for paired native/Context-Q evaluation."""

from __future__ import annotations


def add_paired_checkpoint_epochs(parser, *, default=500_000):
    """Add backwards-compatible and explicit native/context epoch flags."""

    parser.add_argument(
        "--epoch",
        type=int,
        default=default,
        help="Legacy fallback used when a side-specific epoch is omitted.",
    )
    parser.add_argument("--native-epoch", type=int, default=None)
    parser.add_argument("--context-epoch", type=int, default=None)


def paired_checkpoint_epochs(args):
    native_epoch = args.epoch if args.native_epoch is None else args.native_epoch
    context_epoch = args.epoch if args.context_epoch is None else args.context_epoch
    if native_epoch <= 0 or context_epoch <= 0:
        raise ValueError("Checkpoint epochs must be positive")
    return int(native_epoch), int(context_epoch)
