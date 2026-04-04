# AI Robotic Arm Chess System

An intelligent chess system that combines computer vision, digital board rendering, and AI gameplay using Stockfish.

---

## Features

- Chessboard detection using OpenCV  
- Top-view (perspective warp) transformation  
- Digital chess GUI with PNG pieces  
- Stockfish AI integration  
- Legal move highlighting  
- Playable GUI (Player vs AI)  
- Modular scripts for testing and development  

---

## Project Structure


AI ROB ARM/
│── markers/                    # Marker assets (if used)
│── pieces/                     # PNG chess pieces
│── stockfish/                  # Chess engine
│
│── src/
│ ├── detect_board.py           # Detects chessboard
│ ├── detect_main.py            # Detection + top view + GUI (only if board detected)
│ ├── test_chess.py             # Console-based chess test (Stockfish best move)  
│ ├── playable_test.py          # Playable GUI (Player vs AI)
│ ├── test_main.py              # Playable GUI (Player vs AI)
│
│── venv/                       # Virtual environment
│── requirements.txt
│── README.md
│── .gitignore


---

## File Descriptions

### detect_board.py
Detects the chessboard from camera input.

### detect_main.py
- Detects the chessboard  
- Generates top-down (bird’s eye) view  
- Displays digital GUI only when board is detected  

### test_chess.py
- Tests Stockfish integration  
- Runs in console  
- Outputs best move  

### playable_test.py
- Fully playable chess GUI  
- Player (White) vs AI (Stockfish)  
- Click-to-move interaction  
- Legal move highlighting  

### test_main.py
- Combines detection, GUI, and top view  

---

## Setup Instructions

### 1. Clone Repository

git clone <your-repo-link>
cd "AI ROB ARM"


### 2. Create Virtual Environment

python -m venv venv
venv\Scripts\activate


### 3. Install Dependencies

pip install -r requirements.txt


### 4. Add Stockfish

Download Stockfish and place the executable inside:


stockfish/
└── stockfish-windows-x86-64-avx2.exe


---

## How to Run

### Test Stockfish (Console)

python src/test_chess.py


### Play Chess (GUI + AI)

python src/playable_test.py


### Detect Board + Top View + GUI

python src/detect_main.py


### Full System Test

python src/test_main.py


---

## Tech Stack

- Python  
- OpenCV  
- NumPy  
- python-chess  
- Stockfish  

---

## Future Improvements

- Real piece detection  
- Sync physical board with digital board  
- Robotic arm integration  
- Move tracking and history  
- AI/ML enhancements  

---

## Author

Herbert George  

---

## Notes

This project is part of building an AI-powered robotic arm chess system that can detect, analyze, and play chess autonomously.