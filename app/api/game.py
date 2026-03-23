import chess
import torch
import torch.nn.functional as func
#
from ml.encode import encode_board, encode_game, encode_move, decode_move
from ml.model import load_model
#
#This is the main functional of game, the checking whether player can move or not
#And AI predicts its next move


##############################################################
def ai_moves(board: chess.Board, model, device) -> chess.Move:
    state = encode_board(board) #encode it and gets normal board
    tensor = torch.tensor(state).unsqueeze(0).to(device) #(1, 18, 8, 8) and 1 is turn
    with torch.no_grad():
        policy, value = model(tensor)

    probs = func.softmax(policy[0], dim=0)
    best_moves = torch.argsort(probs, descending= True)

    for idx in best_moves:
        move = decode_move(idx.item(), board)
        if move in board.legal_moves:
            return move #return the best possible move

    #return any possible move
    return list(board.legal_moves)[0]
    #if no, will go to gameover
##############################################################


##########################################################################
def player_moves(fen: str, player_move: str, model, device) -> chess.Move:
    board = chess.Board(fen)
    #player moves only in FEN standard

    move = chess.Move.from_uci(player_move)
    if move not in board.legal_moves:
        return None #don't do anything even if player is trying to move illegaly

    board.push(move)
    if board.is_game_over():
        return {
            "move": "",
            "fen": board.fen(),
            "gameover": True,
            "result": board.result()
        } #result of winner move on win
    
    ai_move = ai_moves(board, model, device)
    board.push(ai_move)
    return {
        "move": ai_move.uci(),
        "fen": board.fen(),
        "gameover": board.is_game_over(),
        "result": board.result()
    } #result of ai move
##########################################################################