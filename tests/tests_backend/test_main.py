import pytest
import chess

from api.main import chess as app
from fastapi.testclient import TestClient

'''Tests of endpoints in backend and frontend integrations'''
'''200 - SUCCESSFUL RESPONSE'''
'''400 - BAD REQUEST FROM USER'''

START_FEN = chess.STARTING_FEN
#FEN it is string of board state at moment


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


'''Check of status code 200 from main'''
'''Opening of main page'''
def test_mainpage_open(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


'''Check of successful response on user's input'''
def test_user_legal_move(client: TestClient):
    '''legal move'''
    response = client.post("/move", json={"fen": START_FEN, "move": "e2e4"})
    assert response.status_code == 200
    data = response.json() #response data
    assert "fen" in data
    assert "move" in data
    assert "gameover" in data


'''Check of 400 status code on illegal move'''
def test_user_illegal_move(client: TestClient):
    '''legal move'''
    response = client.post("/move", json={"fen": START_FEN, "move": "e2e8"})
    assert response.status_code == 400
    data = response.json() #response data
    assert data["detail"] == "Illegal move"


'''Response on new game request'''
def test_new_game(client: TestClient):
    response = client.get("/game")
    assert response.status_code == 200
    data = response.json() #response data
    assert "fen" in data
    chess.Board(data["fen"])