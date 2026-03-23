import torch
import chess as ch
#
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
#
from ml.encode import encode_board, encode_game, encode_move, decode_move
from ml.model import load_model
from api.game import player_moves, ai_moves
from api.schemas import MovePlayer, MoveAI
#

chess = FastAPI()
templates = Jinja2Templates(directory="templates")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("weights/model.pt")
model = model.to(device)


#################################################
@chess.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "main.html")
#################################################


###########################################
@chess.post("/move", response_model=MoveAI)
async def make_move(data: MovePlayer):
    result = player_moves(data.fen, data.move, model, device)
    if result is None:
        raise HTTPException(status_code=400, detail="Illegal move")
    return result
###########################################


###################
@chess.get("/game")
async def new_game():
    board = ch.Board()
    return {"fen": board.fen()}
###################


##########################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:chess", host="127.0.0.1", port=8000, reload=True)
    #launch via (venv) in /app uvicorn api.main:chess --reload
##########################