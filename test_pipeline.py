import sys, pathlib
sys.path.append(str(pathlib.Path(r'C:/Users/monic/OneDrive/Desktop/BTP').resolve()))
from src.driver import run_full_pipeline
from pathlib import Path

if __name__ == '__main__':
    text = Path(r'C:/Users/monic/OneDrive/Desktop/BTP/input/lecture.txt').read_text(encoding='utf-8')
    tr, audio = run_full_pipeline(text, 'telugu')
    print('Translated length', len(tr))
    print('Audio path', audio)
