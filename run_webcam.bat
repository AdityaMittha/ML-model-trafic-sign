@echo off
echo ============================================
echo   Traffic Sign Detection - Live Webcam
echo ============================================
echo.
echo Starting webcam with GPU acceleration...
echo Press 'q' in the webcam window to quit.
echo.
cd /d "E:\trafic sign Ml model"
.venv\Scripts\python.exe scripts/inference.py --model "E:\trafic sign Ml model\training\traffic_sign_yolo11n\weights\best.pt" --source 0 --device 0
pause
