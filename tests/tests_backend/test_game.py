import chess
import pytest
from unittest.mock import MagicMock, patch
from api.game import player_moves, ai_moves

'''Test of player_moves response'''

START_FEN = chess.STARTING_FEN
#FEN it is string of board state at moment

@pytest.fixture
def mock_model_and_device():
    device = 'cpu'
    model = MagicMock() #Empty 'AI' that returns what we input

    import torch
    model.return_value = (torch.randn(1, 4672), torch.randn(1, 1)) 
                         #policy head (4672 possible moves) and value head
    return model, device

'''Normal value'''
def test_valid_move_returns_dict(mock_model_and_device):
    model, device = mock_model_and_device
    with patch("api.game.ai_moves", return_value=chess.Move.from_uci("e7e5")):
        result = player_moves(START_FEN, "e2e4", model, device)
    
    assert isinstance(result, dict)
    assert "move" in result
    assert "fen" in result
    assert "gameover" in result

'''Error value'''
def test_invalid_move_returns_none(mock_model_and_device):
    model, device = mock_model_and_device
    #knight can't go e2e5
    result = player_moves(START_FEN, "e2e5", model, device)
    assert result is None


'''Checks for AI always response while field is non-empty'''
def test_ai_always_responds(mock_model_and_device):
    model, device = mock_model_and_device
    with patch("api.game.ai_moves", return_value=chess.Move.from_uci("e7e5")):
        result = player_moves(START_FEN, "e2e4", model, device)
    #AI moved and field is non-empty
    assert result["move"] != "" or result["gameover"] == True


'''Continuous change of FEN after moves'''
def test_fen_changes_after_move(mock_model_and_device):
    model, device = mock_model_and_device
    with patch("api.game.ai_moves", return_value=chess.Move.from_uci("e7e5")):
        result = player_moves(START_FEN, "e2e4", model, device)
    #Check for FEN change
    assert result["fen"] != START_FEN


'''Test of checkmate'''
def test_gameover_on_checkmate(mock_model_and_device):
    #mate in 1 position
    fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 2"
    model, device = mock_model_and_device
    board = chess.Board(fen)
    valid_move = list(board.legal_moves)[0].uci()
    with patch("api.game.ai_moves", return_value=chess.Move.from_uci("d8h4")):  # AI делает мат
        result = player_moves(fen, valid_move, model, device)
    assert "gameover" in result
  

'''As in name, check whether AI return valid inputs'''
def test_ai_returns_legal_move(mock_model_and_device):
    model, device = mock_model_and_device
    ai_response = chess.Move.from_uci("e7e5") #legal move
    with patch("api.game.ai_moves", return_value=ai_response) as mocked:
        player_moves(START_FEN, "e2e4", model, device)
        #is AI was called
        mocked.assert_called_once()
