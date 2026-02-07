#/bin/bash
python3 train.py
uvicorn app.api:app --host "0.0.0.0"
