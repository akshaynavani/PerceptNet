"""Command-line entry point: ``perceptnet <command> ...``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from perceptnet.config import load_config


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", default=None, help="YAML config file")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
        help="override a config value (repeatable), e.g. --set train.lr=1e-3",
    )


def cmd_train(args) -> int:
    from perceptnet.train.trainer import Trainer

    cfg = load_config(args.config, args.overrides)
    trainer = Trainer(cfg)
    result = trainer.train()
    print(json.dumps(result, indent=2))
    return 0


def cmd_export_synthetic(args) -> int:
    from perceptnet.data.synthetic import SyntheticShapes, export_coco

    ds = SyntheticShapes(args.num, (args.height, args.width), seed=args.seed)
    path = export_coco(ds, args.out)
    print(f"wrote {len(ds)} images and {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="perceptnet", description="JAX/XLA perception pipeline")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train / fine-tune a detector")
    _add_config_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("export-synthetic", help="write a synthetic split to disk in COCO format")
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=200)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_export_synthetic)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
