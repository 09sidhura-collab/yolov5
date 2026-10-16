
@echo off
cd /d "%USERPROFILE%\yolov5"
call .venv\Scripts\activate
python vfa_server.py
pause
