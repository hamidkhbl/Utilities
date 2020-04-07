for /F "DELIMS=" %%I IN ('dir /b /s *.zip *.rar') DO (
    "C:\Program Files\7-Zip\7z.exe" x -aos -o"%%~dpI" "%%I"
)