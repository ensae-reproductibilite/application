#/bin/bash
python3 train.py
uvicorn app.api:app --reload --host "0.0.0.0" --port 5000
