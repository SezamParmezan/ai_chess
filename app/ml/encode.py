import chess
import numpy as np
#

PIECES = [chess.PAWN,
          chess.BISHOP,
          chess.KNIGHT,
          chess.ROOK,
          chess.KING,
          chess.QUEEN]


###################################################
def encode_board(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    #12 planes per piece (6 black and 6 white), 0-11
    for i, piece in enumerate(PIECES):
        for sq in board.pieces(piece, chess.WHITE):
            r, c = divmod(sq, 8)
            tensor[i, r, c] = 1.0

        for sq in board.pieces(piece, chess.BLACK):
            r, c = divmod(sq, 8)
            tensor[i + 6, r, c] = 1.0 #cause black on 7th and 8th rows

    #castlings (12-15)
    cr = board.castling_rights
    tensor[12] = float(bool(cr & chess.BB_H1))
    tensor[13] = float(bool(cr & chess.BB_A1))
    tensor[14] = float(bool(cr & chess.BB_H8))
    tensor[15] = float(bool(cr & chess.BB_A8))
    tensor[16] = 1.0 if board.turn == chess.WHITE else 0.0 #16 - 1.0 is white turn and 0.0 is black

    #17 - capture on the way
    if board.ep_square is not None:
        r, c = divmod(board.ep_square, 8)
        tensor[17, r, c] = 1.0

    #18 tensors in total
    return tensor
###################################################


#########################################
def encode_move(move: chess.Move) -> int:
    #encode current move to normal int form
    move_from = move.from_square
    move_to = move.to_square

    #4096 (64 * 64)
    return move_from * 64 + move_to
#########################################


###############################################################
def decode_move(action: int, board: chess.Board) -> chess.Move:
    #reverse of encode_move(), decodes from int back to move
    sq_from = action // 64
    sq_to = action % 64
    move = chess.Move(sq_from, sq_to)

    #checks if pawn on this square to become queen
    if board.piece_type_at(sq_from) == chess.PAWN:
        #pawn becomes queen on 8th row
        if chess.square_rank(sq_to) in (0, 7):
            move = chess.Move(sq_from, sq_to, promotion=chess.QUEEN)

    return move
###############################################################


####################################
def encode_game(game) -> list[dict]:
    board = game.board()
    samples = []
    result = game.headers.get("Result", "*")

    #result of game
    value = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}.get(result, None)
    if value is None:
        return []
    
    #samples with current state data
    for move in game.mainline_moves():
        state = encode_board(board)
        action = encode_move(move)
        samples.append({"state": state, "action": action, "value": value})
        board.push(move)
        value = -value

    return samples
    #it is the grade of the game
####################################