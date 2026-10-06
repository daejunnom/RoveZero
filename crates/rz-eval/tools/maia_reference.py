#!/usr/bin/env python3
"""Generate bounded external LC0 Eigen/protobuf evidence, not ONNX-vs-ONNX.

Run in a separate Python process with LC0's built backends module on PYTHONPATH.
Dependencies: numpy==2.2.6, python-chess==1.999 (chess==1.11.2).
python-chess is an independent fixture oracle, never a product Rules dependency.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import backends
import chess
import numpy as np


CASES = [
    ("start", chess.STARTING_FEN, []),
    ("black-after-e4", chess.STARTING_FEN, ["e2e4"]),
    ("eight-ply-opening", chess.STARTING_FEN,
     "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6".split()),
    ("repeated-start-with-history", chess.STARTING_FEN,
     "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8".split()),
    ("same-board-unknown-prefix", chess.STARTING_FEN.replace("0 1", "8 5"), []),
    ("white-en-passant", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", []),
    ("black-en-passant", "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 2", []),
    ("white-both-castles", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 17 1", []),
    ("black-both-castles", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 18 1", []),
    ("white-all-promotions", "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", []),
    ("black-all-promotions", "4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1", []),
    ("short-fen-history", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 17 1", ["a1a2", "h8h7"]),
]

PROFILES = {
    "maia-1900": (1_262_607, "e2f565f42d7cd9f122557e6dc4eb84e5bbaedceda1d404dc485d3611c7c97a12"),
    "bt4-it332": (382_645_315, "e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618"),
}


def frame(board):
    return dict(pieces=[[int(board.pieces(piece, color)) for piece in range(1, 7)]
                        for color in (chess.WHITE, chess.BLACK)],
                repeated=board.is_repetition(2), en_passant_target=board.ep_square)


def main(args):
    if chess.__version__ != "1.11.2" or np.__version__ != "2.2.6":
        raise ValueError("fixture oracle versions differ")
    commit = subprocess.check_output(["git", "-C", str(args.lc0_source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.lc0_source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if commit != "fd71a2d921b689c5f479d3227c3806c8e272d9c5" or dirty:
        raise ValueError("reference checkout must be the clean pinned LC0 commit")
    if args.output.exists():
        raise ValueError("existing reference evidence is preserved; use a new output path")
    source = args.source.resolve()
    expected_bytes, expected_hash = PROFILES[args.profile]
    if source.stat().st_size != expected_bytes:
        raise ValueError("reference source size differs from selected profile")
    with source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if source_hash != expected_hash:
        raise ValueError("wrong reference source")
    backend = backends.Backend(weights=backends.Weights(str(source)), backend="eigen",
                               options="threads=1,batch_size=16")
    records = []
    for name, fen, moves in CASES:
        board = chess.Board(fen)
        history = [frame(board)]
        for move in moves:
            board.push_uci(move)
            history.append(frame(board))
        game = backends.GameState(fen=fen, moves=moves)
        encoded = game.as_input(backend)
        dense = [encoded.val(plane) if encoded.mask(plane) & (1 << square) else 0.0
                 for plane in range(112) for square in range(64)]
        reference = backend.evaluate(encoded)[0]
        legal_moves = game.moves()
        indices = list(game.policy_indices())
        # Cross-check independent Rules oracle; it does not enter product code.
        if set(legal_moves) != {move.uci() for move in board.legal_moves}:
            raise ValueError(f"legal view disagreement: {name}")
        mapped = []
        for text in legal_moves:
            move = chess.Move.from_uci(text)
            mapped.append(dict(from_square=move.from_square, to_square=move.to_square,
                               promotion=text[4:] or None, castle=board.is_castling(move)))
        logits = list(reference.p_raw(*range(1858)))
        q, draw = reference.q(), reference.d()
        wdl = [(1.0 - draw + q) / 2.0, draw, (1.0 - draw - q) / 2.0]
        legal_logits = np.array(logits, dtype=np.float64)[indices]
        policy = np.exp(legal_logits - legal_logits.max())
        policy /= policy.sum()  # explicit temperature=1, not LC0 search temperature
        # Python binding uses FEN_ONLY. This equals No for start-board prefixes,
        # RepeatOldest otherwise. Keep the chosen C profile explicit per case.
        fill = "no" if fen.split()[0] == chess.STARTING_FEN.split()[0] else "repeat_oldest"
        records.append(dict(name=name, fen=fen, moves=moves, frames=history[-8:][::-1],
                            black_to_move=board.turn == chess.BLACK,
                            castling=[board.has_queenside_castling_rights(chess.WHITE),
                                      board.has_kingside_castling_rights(chess.WHITE),
                                      board.has_queenside_castling_rights(chess.BLACK),
                                      board.has_kingside_castling_rights(chess.BLACK)],
                            halfmove_clock=board.halfmove_clock, history_fill=fill,
                            input=dense, legal_moves=mapped, indices=indices,
                            policy_logits=logits, wdl=wdl, legal_policy=policy.tolist()))
    module = Path(backends.__file__)
    with module.open("rb") as stream:
        module_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    document = dict(schema=1, reference="lc0-v0.32.1-eigen-original-protobuf",
                    reference_commit=commit,
                    reference_module_sha256=module_hash, source_sha256=source_hash,
                    fixture_oracle="python-chess-1.999/chess-1.11.2", cases=records)
    args.output.write_text(json.dumps(document, separators=(",", ":")) + "\n")
    print(json.dumps(dict(cases=len(records), output=str(args.output),
                         reference_module_sha256=module_hash)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--lc0-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="maia-1900")
    main(parser.parse_args())
