"""Command-line entry point for the offline release workflow.

Examples (all local; nothing is emailed)::

    python -m release explain  --counts data/counts.csv --policy data/policy.json \
        --published data/published.csv --period 2026-03 --outbox /tmp/outbox
    python -m release prepare  --period 2026-03 --recipient partner-a
    python -m release approve  --material m2 --editor alice
    python -m release schedule --approval a1 --due 2026-04-05T09:00:00+08:00
    python -m release send     --approval a1 --delivered yes
    python -m release manifest --out manifest.json

The ``prepare``/``approve``/``schedule``/``send`` steps operate on a persisted
book so that approval and dispatch are separate, auditable moments.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m release")
    parser.add_argument("--book", default="release-book.pkl",
                        help="local workflow book (pickle, kept offline)")
    parser.add_argument("--counts", default="data/counts.csv")
    parser.add_argument("--policy", default="data/policy.json")
    parser.add_argument("--published", default="data/published.csv")
    parser.add_argument("--receipts", default=None)
    parser.add_argument("--outbox", default="outbox")
    sub = parser.add_subparsers(dest="command", required=True)

    explain = sub.add_parser("explain", help="value-free publishability report")
    explain.add_argument("--period", required=True)
    explain.add_argument("--metrics", nargs="+", default=["participants"])
    explain.add_argument("--audience", default="external-public")
    explain.add_argument("--recipient", action="append", default=[])
    explain.add_argument("--alternatives", action="store_true")
    explain.add_argument("--out")

    prepare = sub.add_parser("prepare", help="create a period sheet")
    prepare.add_argument("--period", required=True)
    prepare.add_argument("--metrics", nargs="+", default=["participants"])
    prepare.add_argument("--recipient", action="append", default=[])
    prepare.add_argument("--audience", default="external-public")
    prepare.add_argument("--course-group", action="append", default=[])

    approve = sub.add_parser("approve")
    approve.add_argument("--material", required=True)
    approve.add_argument("--editor", required=True)

    schedule = sub.add_parser("schedule")
    schedule.add_argument("--approval", required=True)
    schedule.add_argument("--due", required=True)

    send = sub.add_parser("send")
    send.add_argument("--approval", required=True)
    send.add_argument("--delivered", default="unknown")
    send.add_argument("--proof-unsent", action="store_true")

    withdraw = sub.add_parser("withdraw-schedule")
    withdraw.add_argument("--schedule", required=True)
    withdraw.add_argument("--reason", required=True)

    retract = sub.add_parser("retract")
    retract.add_argument("--material", required=True)
    retract.add_argument("--reason", required=True)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    book = Path(args.book)
    workflow = _load(args, book)

    if args.command == "explain":
        material = workflow.prepare_period_material(
            period=args.period, metrics=args.metrics,
            audience=args.audience, recipients=args.recipient,
            release_id=f"{args.period}-explain")
        report = workflow.explain_material(material)
        if args.alternatives:
            report["alternatives"] = {
                metric: workflow.alternatives(material, metric)
                for metric in args.metrics
            }
        _emit(report, args.out)
        _save(workflow, book)
        return 0 if report["releasable"] or not report["conflicts"] else 1

    if args.command == "prepare":
        groups = [_split(group) for group in args.course_group] or None
        material = workflow.prepare_period_material(
            period=args.period, metrics=args.metrics,
            audience=args.audience, recipients=args.recipient,
            course_groups=groups)
        _emit(material.manifest(), None)
        _save(workflow, book)
        return 0

    if args.command == "approve":
        approval = workflow.approve(workflow.materials[args.material],
                                    editor=args.editor)
        _emit(approval.manifest(), None)
        _save(workflow, book)
        return 0

    if args.command == "schedule":
        schedule = workflow.schedule(workflow.approvals[args.approval],
                                     due_at=args.due)
        _emit(schedule.manifest(), None)
        _save(workflow, book)
        return 0

    if args.command == "send":
        receipt = workflow.send(workflow.approvals[args.approval],
                                delivered=args.delivered,
                                proof_unsent=args.proof_unsent)
        _emit(receipt.__dict__, None)
        _save(workflow, book)
        return 0

    if args.command == "withdraw-schedule":
        workflow.withdraw_schedule(workflow.schedules[args.schedule], reason=args.reason)
        _save(workflow, book)
        return 0

    if args.command == "retract":
        workflow.retract(workflow.materials[args.material], reason=args.reason)
        _save(workflow, book)
        return 0

    if args.command == "manifest":
        if args.out:
            workflow.export_manifest(args.out)
            print(args.out)
        else:
            _emit(workflow.delivery_manifest(), None)
        return 0

    return 2


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def _load(args, book: Path):
    from .workflow import Workflow

    if book.exists():
        with book.open("rb") as stream:
            return pickle.load(stream)
    counts = Path(args.counts)
    policy = Path(args.policy)
    published = Path(args.published) if args.published else None
    repository = Path(__file__).resolve().parents[2]
    if not counts.exists():
        counts = repository / "data" / "counts.csv"
    if not policy.exists():
        policy = repository / "data" / "policy.json"
    if published is not None and not published.exists():
        published = repository / "data" / "published.csv"
    return Workflow.from_files(
        counts=counts, policy=policy, published=published,
        receipts=args.receipts, outbox=args.outbox)


def _save(workflow, book: Path) -> None:
    with book.open("wb") as stream:
        pickle.dump(workflow, stream)


def _emit(payload, destination: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    if destination:
        Path(destination).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    raise SystemExit(main())
