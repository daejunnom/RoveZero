#!/usr/bin/env python3
"""Synthetic scripted legal-move UCI fixture; no engine/search/GPU proof."""
import sys
moves = []
line_moves = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'g8f6', 'd2d3', 'f8e7']
for raw in sys.stdin:
    command = raw.strip()
    if command == 'uci':
        print('id name E02SyntheticFixture')
        print('id author RoveZero test fixture')
        print('option name Threads type spin default 1 min 1 max 1')
        print('option name Hash type spin default 1 min 1 max 16')
        print('option name Ponder type check default false')
        print('option name Seed type spin default 1 min 1 max 10000')
        print('uciok', flush=True)
    elif command == 'isready':
        print('readyok', flush=True)
    elif command.startswith('position '):
        moves = command.split(' moves ', 1)[1].split() if ' moves ' in command else []
    elif command.startswith('go '):
        n = len(moves)
        print('info depth 1 score cp 0 nodes 1 time 0')
        print('bestmove ' + (line_moves[n] if n < len(line_moves) else '0000'), flush=True)
    elif command == 'quit':
        break
