import pytest

from playwright.sync_api import Page
from typing import Generator

'''Test of frontend features on main page'''
'''Functional'''

BASE_URL = "http://127.0.0.1:8000"


'''Test if player can move legally'''
def test_player_move_legal(page: Page, server):
    page.goto(BASE_URL)

    page.locator(".sq[data-idx='12']").click() #e2
    page.locator(".sq[data-idx='28']").click() #e4
    page.wait_for_timeout(5000) #Wait for AI response, if it has, then move was performed
    history = page.locator("#history")
    assert history.inner_text() != ""


'''Test if player cannot move illegally'''
def test_player_move_legal(page: Page, server):
    page.goto(BASE_URL)

    page.locator(".sq[data-idx='12']").click() #e2
    page.locator(".sq[data-idx='60']").click() #e4
    page.wait_for_timeout(5000) #Wait for AI response, if it has, then move was performed
    history = page.locator("#history")
    assert history.inner_text() == ""


'''Test for flip board'''
def test_flip_board(page: Page, server):
    page.goto(BASE_URL)

    coords_before = page.locator("#file-coords-bot .coord").nth(0).inner_text()
    page.locator("button", has_text="Flip Board").click()
    coords_after = page.locator("#file-coords-bot .coord").nth(0).inner_text()
    assert coords_before != coords_after


'''Test if new game button clears history'''
def test_clear_at_newgame(page: Page, server):
    page.goto(BASE_URL)

    #any move
    page.locator(".sq[data-idx='12']").click()
    page.locator(".sq[data-idx='28']").click()
    page.wait_for_timeout(4000)

    #New game
    page.locator("button", has_text="New Game").click()
    page.wait_for_timeout(500)
    assert page.locator("#history").inner_text() == ""


'''Test for message after checkmate'''
def test_after_checkmate(page: Page, server):
    page.goto(BASE_URL)

    #Mate situation
    page.evaluate("() => { document.getElementById('overlay').classList.add('show'); }")
    
    assert page.locator("#overlay").is_visible()