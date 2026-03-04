"""봇 + Streamlit 대시보드 동시 실행 런처"""
import subprocess
import sys
import os
import time
import signal
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = Path(PROJECT_ROOT) / "logs"


def main():
    processes = []

    print()
    print("=" * 60)
    print("  🚀 AI 선물 트레이딩 봇 + 대시보드 실행")
    print("=" * 60)
    print()

    try:
        # 1. 봇 시작 (main.py의 대화형 메뉴)
        print("[1/2] 트레이딩 봇 시작...")
        bot_proc = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=PROJECT_ROOT
        )
        processes.append(('봇', bot_proc))

        # 봇 초기화 대기
        print("       봇 초기화 대기 중 (3초)...")
        time.sleep(3)

        # 2. 대시보드 시작 (stdout/stderr를 로그 파일로 리디렉션)
        print("[2/2] Streamlit 대시보드 시작...")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        dashboard_log_path = LOG_DIR / "dashboard.log"
        dashboard_log = open(dashboard_log_path, "a", encoding="utf-8")
        dashboard_proc = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run",
                os.path.join("streamlit_dashboard", "app.py"),
                "--server.port", "8501",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ],
            cwd=PROJECT_ROOT,
            stdout=dashboard_log,
            stderr=dashboard_log,
        )
        processes.append(('대시보드', dashboard_proc))
        print(f"       대시보드 로그: {dashboard_log_path}")

        print()
        print("━" * 60)
        print("  ✅ 시스템 실행 중")
        print("  📊 대시보드: http://localhost:8501")
        print("  💡 Ctrl+C로 전체 종료")
        print("━" * 60)
        print()

        # 프로세스 감시
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n⚠️ {name} 프로세스가 종료되었습니다 (코드: {proc.returncode})")
                    raise KeyboardInterrupt
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n⏹️ 시스템 종료 중...")
    finally:
        for name, proc in processes:
            if proc.poll() is None:
                print(f"  {name} 종료 중...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        # 대시보드 로그 파일 핸들 정리
        try:
            dashboard_log.close()
        except Exception:
            pass

        print("✅ 전체 시스템 종료 완료")


if __name__ == '__main__':
    main()
