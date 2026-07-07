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


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _build_engine(args, batch_size: int | None = None):
    from perceptnet.inference.engine import InferenceEngine

    cfg = load_config(args.config, args.overrides)
    if not cfg.data.class_names:
        raise SystemExit("data.class_names must be set for inference (use the run's config.yaml)")
    return cfg, InferenceEngine.from_checkpoint(cfg, args.checkpoint, batch_size=batch_size)


def cmd_infer(args) -> int:
    """Run detection on an image, a directory of images, a video file or a webcam index."""
    import time
    from pathlib import Path

    import cv2

    from perceptnet.inference.visualize import draw_detections

    cfg, engine = _build_engine(args)
    engine.warmup()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = cfg.data.class_names
    src = Path(args.source)

    if args.source.isdigit() or src.suffix.lower() in _VIDEO_EXTS:
        cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else str(src))
        if not cap.isOpened():
            raise SystemExit(f"could not open video source {args.source}")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = None
        frames = 0
        t_start = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and frames >= args.max_frames):
                break
            t0 = time.perf_counter()
            dets = engine.detect([frame], args.score_threshold)[0]
            fps = 1.0 / max(time.perf_counter() - t0, 1e-9)
            vis = draw_detections(frame, dets, names, fps=fps)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
            writer.write(vis)
            if args.show:
                cv2.imshow("PerceptNet", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frames += 1
        cap.release()
        if writer is not None:
            writer.release()
        elapsed = time.perf_counter() - t_start
        print(json.dumps({"frames": frames, "avg_fps": frames / max(elapsed, 1e-9), **engine.timings.summary()}, indent=2))
        return 0

    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in _IMAGE_EXTS) if src.is_dir() else [src]
    results = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"skipping unreadable {path}", file=sys.stderr)
            continue
        dets = engine.detect([frame], args.score_threshold)[0]
        cv2.imwrite(str(out_dir / path.name), draw_detections(frame, dets, names))
        results.append({
            "image": str(path),
            "detections": [
                {"box": [round(float(v), 2) for v in b], "score": round(float(s), 4), "label": names[int(c)]}
                for b, s, c in zip(dets.boxes, dets.scores, dets.labels)
            ],
        })
    (out_dir / "detections.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"images": len(results), "out": str(out_dir), **engine.timings.summary()}, indent=2))
    return 0


def cmd_export_hlo(args) -> int:
    """Dump the StableHLO and XLA-optimized HLO of the compiled inference graph."""
    _, engine = _build_engine(args, batch_size=args.batch_size)
    for path in engine.dump_hlo(args.out):
        print(f"wrote {path}")
    print(json.dumps({"compile_time_s": engine.compile_time_s, **engine.cost_analysis()}, indent=2))
    return 0


def cmd_eval(args) -> int:
    from perceptnet.eval.evaluate import format_report, run_evaluation

    cfg = load_config(args.config, args.overrides)
    report = run_evaluation(cfg, args.checkpoint, split=args.split, out=args.out)
    print(format_report(report))
    return 0


def cmd_bench(args) -> int:
    from pathlib import Path

    from perceptnet.bench.timing import format_table

    results, extra = [], {}
    if args.suite in ("ops", "all"):
        from perceptnet.bench.ops_bench import run_all

        results += run_all(quick=args.quick)
    if args.suite in ("latency", "all"):
        from perceptnet.bench.latency_bench import bench_latency
        from perceptnet.train.checkpoint import load_inference_variables

        cfg = load_config(args.config, args.overrides)
        variables = load_inference_variables(args.checkpoint) if args.checkpoint else None
        batch_sizes = tuple(int(b) for b in args.batch_sizes.split(","))
        rows, summary = bench_latency(
            cfg, variables, batch_sizes, iters=5 if args.quick else 30, include_eager=not args.no_eager
        )
        results += rows
        extra["latency_summary"] = summary
    print(format_table(results))
    if extra.get("latency_summary"):
        print(json.dumps(extra["latency_summary"], indent=2))
    if args.out:
        import jax

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "jax_backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "results": [r.row() for r in results],
            **extra,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
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

    p = sub.add_parser("infer", help="run detection on images, a folder, a video or a webcam")
    _add_config_args(p)
    p.add_argument("--checkpoint", required=True, help="best_model.msgpack / final_model.msgpack")
    p.add_argument("--source", required=True, help="image, directory, video file or webcam index")
    p.add_argument("--out", default="outputs/infer")
    p.add_argument("--score-threshold", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--show", action="store_true", help="display frames in a window")
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("eval", help="evaluate mAP and P/R/F1 across score thresholds")
    _add_config_args(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--out", default="outputs/eval.json")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("bench", help="benchmark numeric ops and end-to-end latency")
    _add_config_args(p)
    p.add_argument("--suite", default="all", choices=["ops", "latency", "all"])
    p.add_argument("--checkpoint", default=None, help="optional weights for the latency suite")
    p.add_argument("--batch-sizes", default="1,4,8")
    p.add_argument("--quick", action="store_true", help="fewer sizes and iterations")
    p.add_argument("--no-eager", action="store_true", help="skip the slow eager baseline")
    p.add_argument("--out", default="outputs/bench.json")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("export-hlo", help="dump StableHLO / optimized HLO of the inference graph")
    _add_config_args(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--out", default="outputs/hlo")
    p.set_defaults(func=cmd_export_hlo)

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
