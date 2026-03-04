@echo off
chcp 65001 >nul
echo ============================================
echo   학습 데이터 초기화 (풀 파이프라인 준비)
echo ============================================
echo.

set ROOT=%~dp0

:: 모델 체크포인트 삭제
echo [1/4] 모델 체크포인트 삭제...
del /q "%ROOT%\models\*.pt" 2>nul
echo       완료

:: 캔들 데이터 삭제
:: echo [2/4] 캔들 데이터 삭제...
:: del /q "%ROOT%\data\raw\*.csv" 2>nul
:: echo       완료

:: 백테스트 결과 삭제
echo [3/4] 백테스트 결과 삭제...
del /q "%ROOT%\backtesting\results\*.csv" 2>nul
echo       완료

:: 로그 초기화
echo [4/4] 로그 초기화...
del /q "%ROOT%\logs\*.log" 2>nul
echo       완료

echo.
echo ============================================
echo   초기화 완료! 풀 파이프라인 실행 가능
echo ============================================
pause
