### pydantic validation of moves
from pydantic import BaseModel
#

#############################
class MovePlayer(BaseModel):
    fen: str #current player's figure position in FEN format (Forsyth–Edwards Notation)
    move: str #player move in UCI format (e2e4 jokes)
#############################


########################
class MoveAI(BaseModel):
    fen: str #after ai moved figure position in FEN format also
    move: str #ai move in UCI format
    gameover: bool #checkmate bool
    result: str | None # "1-0" or "0-1" or "1/2-1/2"
########################