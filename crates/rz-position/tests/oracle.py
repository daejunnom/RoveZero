#!/usr/bin/env python3
"""Independent, test-only standard-chess oracle; never a product dependency.

Install only into a test environment:
    python -m pip install python-chess==1.999 chess==1.11.2

The GPL-3.0-or-later python-chess implementation runs as a separate test process.
No external implementation source is copied into the Rust engine. Output uses a
bounded TSV protocol so the Rust test needs no JSON or Python runtime dependency.
"""

import argparse
import importlib.metadata
from pathlib import Path
import random
import sys

import chess


EXPECTED_VERSIONS = {"python-chess": "1.999", "chess": "1.11.2"}
SPECIAL_POSITIONS = [
    ("both_castles", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
    ("castle_through_attack", "r3kr1r/8/8/8/8/8/8/R3K2R w KQ - 0 1"),
    ("ep_discovered_check", "4k3/8/8/r4pPK/8/8/8/8 w - f6 0 2"),
    ("ep_escapes_check", "4k3/8/8/3pP3/4K3/8/8/8 w - d6 0 2"),
    ("double_check", "k3r3/8/8/8/1b6/8/7R/4K3 w - - 0 1"),
    ("pinned_rook", "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"),
    ("promotion_capture", "1r2k3/P7/8/8/8/8/7p/R3K3 w Q - 0 1"),
    ("checkmate", "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1"),
    ("stalemate", "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"),
]


def emit(*fields):
    print("\t".join(str(field) for field in fields))


def perft(board, depth):
    if depth == 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    nodes = 0
    for move in list(board.legal_moves):
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes


def public_positions():
    fixture = Path(__file__).with_name("fixtures") / "perft.tsv"
    for line in fixture.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, fen, *_counts = line.split("\t")
            yield name, fen


def trace(name, fen, rng, plies, all_initial_edges, divide_depth=None):
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError(f"Invalid oracle fixture: {name}: {fen}")
    emit("BEGIN", name, board.fen(en_passant="fen"))
    path = []
    for ply in range(plies + 1):
        legal = sorted(board.legal_moves, key=lambda move: move.uci())
        emit(
            "POSITION", name, ",".join(path), board.fen(en_passant="fen"),
            ",".join(move.uci() for move in legal), int(board.is_check()),
            int(board.is_checkmate()), int(board.is_stalemate()),
        )
        if ply == 0 and all_initial_edges:
            for move in legal:
                board.push(move)
                emit("EDGE", name, ",".join(path), move.uci(), board.fen(en_passant="fen"))
                if divide_depth is not None:
                    emit("DIVIDE", name, move.uci(), divide_depth, perft(board, divide_depth - 1))
                board.pop()
        if not legal or ply == plies:
            break
        move = rng.choice(legal)
        board.push(move)
        emit("SELECT", name, ",".join(path), move.uci(), board.fen(en_passant="fen"))
        path.append(move.uci())
    emit("END", name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--plies", type=int, default=96)
    parser.add_argument("--divide-fen")
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()
    for package, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            parser.error(f"{package}=={expected} required; found {actual}")
    if args.divide_fen is not None:
        if not 1 <= args.depth <= 4:
            parser.error("divide depth must be between 1 and 4")
        board = chess.Board(args.divide_fen)
        if not board.is_valid():
            parser.error("invalid divide FEN")
        for move in sorted(board.legal_moves, key=lambda move: move.uci()):
            board.push(move)
            emit(move.uci(), perft(board, args.depth - 1))
            board.pop()
        return
    if not 1 <= args.games <= 32 or not 1 <= args.plies <= 256:
        parser.error("games must be 1..32 and plies must be 1..256")
    emit("ORACLE", "python-chess", "1.999", "chess", "1.11.2", args.seed)
    rng = random.Random(args.seed)
    for name, fen in public_positions():
        trace(name, fen, rng, 8, True, divide_depth=3)
    for name, fen in SPECIAL_POSITIONS:
        trace(name, fen, rng, 8, True)
    for game in range(args.games):
        trace(f"random_{game}", chess.STARTING_FEN, rng, args.plies, False)


if __name__ == "__main__":
    main()
