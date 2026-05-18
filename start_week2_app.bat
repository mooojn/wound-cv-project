@echo off
title Wound CV Project - Week 2 App Starter
echo =============================================================
echo             WOUND CV PROJECT - APP LAUNCHER                  
echo =============================================================
echo.

:: 1. Start Python Backend Classification Server
echo [1/2] Starting Python Flask API Server (In background)...
start "Wound CV Flask Server" cmd /k "python week2/app_server.py"
timeout /t 2 >nul

:: 2. Start React Frontend Vite Server
echo [2/2] Starting React Frontend Vite Server (In background)...
cd app
start "Wound CV Vite Frontend" cmd /k "npm run dev"
timeout /t 2 >nul

echo.
echo =============================================================
echo   All servers started successfully!
echo   - Backend API:  http://localhost:5000
echo   - Frontend App: http://localhost:5173 (or http://localhost:5174)
echo.
echo   Opening browser now...
echo =============================================================
echo.

timeout /t 1 >nul
:: Open default web browser to local app
start http://localhost:5173

exit
