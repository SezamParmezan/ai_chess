# AI chess

A chess AI trained with public dataset from [Lichess](https://database.lichess.org/) games. A simple singleplayer game with AI chess! This is my open-sourced pet-project, feel free to share and rate it

---

# How it works

The model is a ResNet with two heads — a **policy head** that predicts the next move, and a **value head** that estimates who's winning. It was trained on Lichess PGN data encoded as 18-plane 8×8 tensors.

The interface is a website powered by FastAPI with HTML/CSS/JS with no external UI frameworks or libs.

---

## Requirements

- Python 3.10+
- requirements.txt
You don't even need to install it manually, just launch "start" version for your OS!

---

## Start

**Windows** — just double-click:
```
start.bat
```

**Mac / Linux:**
```bash
chmod +x start.sh
./start.sh
```

The 'start' will:
1. Create a virtual environment on first run
2. Install all dependencies from `requirements.txt`
3. Open `http://127.0.0.1:8000` in your browser
4. Start the server
 
---

## Manual start

If you want to launch it manually, short command is available at the end of `main.py` file!

1. cd app
2. Create a virtual environment with python -m venv .venv
3. 
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

4. pip install -r requirements.txt
5. uvicorn api.main:chess --host 127.0.0.1 --port 8000
6. Open `http://127.0.0.1:8000` and play

---

## Project structure

```
ai_chess/
├── app/
│   ├── api/
│   │   ├── main.py        # FastAPI app, routes
│   │   ├── game.py        # move validation, AI move selection
│   │   └── schemas.py     # Pydantic models
│   ├── ml/
│   │   ├── dataset.py     # reads .pgn.zst files
│   │   ├── encode.py      # board → tensor, move → int
│   │   ├── model.py       # ResNet architecture
│   │   └── train.py       # training loop
│   ├── templates/
│   │   └── main.html      # chess UI
│   └── weights/
│       └── model.pt       # last version of trained model weights
├── colab/
│   └── train.ipynb        # training notebook for Google Colab
├── start.bat              # Windows launcher
├── start.sh               # Mac/Linux launcher
└── requirements.txt       # list of required libs and frameworks
```

---

## Train by your own

The presented model is based on 15,000 games and 30 epochs, which you think is not enough for you, so here's the instruction for your own model

1. Upload a Lichess dataset (`.pgn.zst`) to Google Drive from [database.lichess.org](https://database.lichess.org)
2. Open `colab/train.ipynb` in Google Colab (You can install Google Colab extension in VS Code for easier work)
3. Go set core like 'core' -> create new server -> GPU -> T4 -> latest version -> Python 3
4. Run all cells one by one
5. Download `model.pt` from Google Drive and place it in `app/weights/`. Don't forget to remove the previous one!
6. Perconally I've used 15,000 games and 30 epochs, but you can enter more until reach the bottleneck of Google Colab powers

---

## Dependencies

Main and most used

| Package | Purpose |
|---|---|
| `torch` | neural network |
| `chess` | move validation, PGN parsing |
| `zstandard` | reading `.pgn.zst` datasets |
| `fastapi` | web server |
| `uvicorn` | ASGI server |
| `jinja2` | HTML templating |
| `numpy` | tensor operations |

---

## Notes

- The AI always promotes to queen (hardcoded for simplicity)
- On CPU it thinks for a second or two — normal, no GPU needed to play
- The model plays legal moves but isn't Magnus Carlsen — it's trained on ~15k games with 30 epochs, which is a starting point not a finished engine. I even beat him twice!