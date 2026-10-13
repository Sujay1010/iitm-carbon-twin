@echo off
cd /d "%~dp0"
py -m streamlit run app.py
if errorlevel 1 python -m streamlit run app.py
pause
