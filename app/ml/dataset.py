import io
import chess.pgn
import zstandard as zstd #lib to read .zst files
from contextlib import contextmanager

#I love to call functions like toDo, openZST but it will not be clean code

############################
@contextmanager
def open_ZST(zst_path):
    with open(zst_path, "rb") as f:
        #We need to decompress .zst to .pgn
        decomp = zstd.ZstdDecompressor()
        reader = decomp.stream_reader(f)
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')

        try:
            yield text_stream
        finally:
            text_stream.close()
############################


############################
def load_games(pgnFILE, max_games = 10000):
    #pgnFILE is the return from open_ZST function, clear thing
    games = []

    with open_ZST(pgnFILE) as f:
        for _ in range(max_games):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)

    return games
############################