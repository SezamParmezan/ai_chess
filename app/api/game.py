import chess
import torch
import torch.nn.functional as func
import sys
sys.path.append("app/ml")
#
from ml.encode import encode_board, encode_game, encode_move
from ml.model import load_model
#
#This is the main functional of game, the checking whether player can move or not
#And AI predicts its next move

