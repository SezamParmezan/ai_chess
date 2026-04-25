import pytest

from playwright.sync_api import Page
from typing import Generator

'''Test of frontend features on main page'''
'''Cosmetic'''

BASE_URL = "http://127.0.0.1:8000"


'''Test of title'''
def test_title(page: Page, server):
    page.goto(BASE_URL)

    title = page.locator("h1")
    assert title.is_visible()
    assert title.inner_text() == "AI Chess"


'''Chessboard - 64 squares'''
def test_64_squares(page: Page, server):
    page.goto(BASE_URL)

    squares_num = page.locator(".sq")
    assert squares_num.count() == 64


'''Chessboard - each squares has equal size'''
def test_square_size(page: Page, server):
    page.goto(BASE_URL)

    squares = page.locator(".sq")
    sizes = set()
    for i in range(64):
        square = squares.nth(i).bounding_box()
        sizes.add((round(square["width"]), round(square["height"])))

    assert len(sizes) == 1


'''Chessboard - check if ranks (a-h) are correct and shown'''
def test_ranks_columns(page: Page, server):
    page.goto(BASE_URL)

    expected_ranks = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    #All there
    rank_coords_top = page.locator("#file-coords-top .coord")
    assert rank_coords_top.count() == 8

    #what we see and compare to real ranks in frontend
    actual_ranks = [rank_coords_top.nth(i).inner_text().strip() for i in range(8)]
    assert actual_ranks == expected_ranks, f"Expected {expected_ranks}, got {actual_ranks}"

    for idx, letter in enumerate(expected_ranks):
        #Center X of each column
        coord_box = rank_coords_top.nth(idx).bounding_box()
        coord_center_x = coord_box["x"] + coord_box["width"] / 2

        #Column squares: idx % 8 == idx (a=0 ... h=7)
        col_square = page.locator(f'.sq[data-idx="{idx}"]')
        sq_box = col_square.bounding_box()
        sq_center_x = sq_box["x"] + sq_box["width"] / 2

        assert abs(coord_center_x - sq_center_x) < 4, (
            f"Rank '{letter}' label center ({coord_center_x:.1f}) "
            f"does not align with column square center ({sq_center_x:.1f})"
        )


'''Chessboard - check if ranks (1-8) are correct and shown'''
def test_ranks_rows(page: Page, server):
    page.goto(BASE_URL)

    expected_ranks = ['8', '7', '6', '5', '4', '3', '2', '1']

    #All there
    rank_coords_row = page.locator("#rank-coords .rank")
    assert rank_coords_row.count() == 8

    #what we see and compare to real ranks in frontend
    actual_ranks = [rank_coords_row.nth(i).inner_text().strip() for i in range(8)]
    assert actual_ranks == expected_ranks, f"Expected {expected_ranks}, got {actual_ranks}"

    for idx, num in enumerate(expected_ranks):
        #Center Y of the rank label
        coord_box = rank_coords_row.nth(idx).bounding_box()
        coord_center_y = coord_box["y"] + coord_box["height"] / 2

        #Row squares: rank "8" = idx 56-63 (top row), "1" = idx 0-7 (bottom row)
        rank_number = int(num)
        row_start_idx = (rank_number - 1) * 8  # leftmost square of that rank
        row_square = page.locator(f'.sq[data-idx="{row_start_idx}"]')
        sq_box = row_square.bounding_box()
        sq_center_y = sq_box["y"] + sq_box["height"] / 2

        assert abs(coord_center_y - sq_center_y) < 4, (
            f"Rank '{num}' label center Y ({coord_center_y:.1f}) "
            f"does not align with row square center Y ({sq_center_y:.1f})"
        )


'''Chessboard - all pieces at their place at game start'''
def test_initial_pieces(page: Page, server):
    page.goto(BASE_URL)

    expected_pieces = {
        # White pieces (rank 1, idx 0-7)
        0: '♖', 1: '♘', 2: '♗', 3: '♕', 4: '♔', 5: '♗', 6: '♘', 7: '♖',
        # White pawns (rank 2, idx 8-15)
        8: '♙', 9: '♙', 10: '♙', 11: '♙', 12: '♙', 13: '♙', 14: '♙', 15: '♙',
        # Black pawns (rank 7, idx 48-55)
        48: '♟', 49: '♟', 50: '♟', 51: '♟', 52: '♟', 53: '♟', 54: '♟', 55: '♟',
        # Black pieces (rank 8, idx 56-63)
        56: '♜', 57: '♞', 58: '♝', 59: '♛', 60: '♚', 61: '♝', 62: '♞', 63: '♜',
    }

    for idx, expected_symbol in expected_pieces.items():
        square = page.locator(f'.sq[data-idx="{idx}"]')
        piece = square.locator('.piece-unicode')

        assert piece.count() == 1, f"Expected a piece at square {idx}, found none"
        actual_symbol = piece.inner_text().strip()
        assert actual_symbol == expected_symbol, (
            f"Square {idx}: expected '{expected_symbol}', got '{actual_symbol}'"
        )

    # Ranks 3-6 (idx 16-47) must be empty
    for idx in range(16, 48):
        square = page.locator(f'.sq[data-idx="{idx}"]')
        piece = square.locator('.piece-unicode')
        assert piece.count() == 0, f"Square {idx} should be empty but has a piece"