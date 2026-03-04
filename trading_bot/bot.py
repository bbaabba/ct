"""통합 자동 거래 봇"""
import time
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta

from data.binance_client import BinanceClient
from data.collectors.historical_collector import HistoricalDataCollector
from strategies.base import BaseStrategy, Signal
from strategies.technical.ma_crossover import MACrossoverStrategy
from strategies.technical.rsi_strategy import RSIStrategy
from strategies.technical.bollinger_bands import BollingerBandsStrategy
from regime.market_regime import RegimeDetector
from strategy_selector.performance_tracker import PerformanceTracker
from strategy_selector.strategy_selector import StrategySelector
from execution.risk_manager import RiskManager, RiskLimits
from utils.logger import setup_logger

logger = setup_logger(__name__)


class TradingBot:
    """
    통합 자동 거래 봇

    전체 시스템을 통합하여 자동으로 거래를 실행합니다.
    """

    def __init__(
        self,
        symbol: str = 'BTCUSDT',
        interval: str = '1h',
        initial_balance: float = 10000.0,
        testnet: bool = True,
        paper_trading: bool = True
    ):
        """
        Args:
            symbol: 거래 심볼
            interval: 시간 간격
            initial_balance: 초기 자본
            testnet: 테스트넷 사용 여부
            paper_trading: 페이퍼 트레이딩 여부 (실제 주문 X)
        """
        self.symbol = symbol
        self.interval = interval
        self.initial_balance = initial_balance
        self.paper_trading = paper_trading

        # Binance 클라이언트
        self.client = BinanceClient(testnet=testnet)
        self.collector = HistoricalDataCollector(self.client)

        # 전략 초기화
        self.strategies: Dict[str, BaseStrategy] = {
            'MA_Crossover': MACrossoverStrategy(),
            'RSI': RSIStrategy(),
            'BollingerBands': BollingerBandsStrategy()
        }

        # 시스템 컴포넌트 초기화
        self.regime_detector = RegimeDetector()
        self.performance_tracker = PerformanceTracker()
        self.strategy_selector = StrategySelector(
            strategies=self.strategies,
            regime_detector=self.regime_detector,
            performance_tracker=self.performance_tracker,
            params={
                'regime_weight': 0.5,
                'performance_weight': 0.5,
                'min_trades_for_perf': 5,
                'rebalance_interval': 24
            }
        )

        # 리스크 관리
        self.risk_manager = RiskManager(
            initial_balance=initial_balance,
            limits=RiskLimits(
                max_position_size_pct=0.2,
                max_daily_loss_pct=0.05,
                stop_loss_pct=0.02,
                take_profit_pct=0.04
            )
        )

        # 상태 변수
        self.is_running = False
        self.last_check_time = None
        self.current_position = None
        self.cycle_count = 0

        logger.info(f"TradingBot 초기화 완료")
        logger.info(f"  심볼: {symbol}, 간격: {interval}")
        logger.info(f"  초기 자본: {initial_balance:.2f} USDT")
        logger.info(f"  페이퍼 트레이딩: {paper_trading}")

    def get_latest_data(self, days: int = 14) -> pd.DataFrame:
        """최신 시장 데이터 가져오기"""
        try:
            df = self.collector.get_recent_data(
                symbol=self.symbol,
                interval=self.interval,
                days=days
            )
            logger.info(f"데이터 수집 완료: {len(df)}개")
            return df
        except Exception as e:
            logger.error(f"데이터 수집 실패: {e}")
            return None

    def check_stop_loss_take_profit(self):
        """Stop-Loss / Take-Profit 확인"""
        if self.current_position is None:
            return

        try:
            # 현재 가격 조회
            current_price = self.client.get_symbol_price(self.symbol)

            # SL/TP 확인
            action = self.risk_manager.check_stop_loss_take_profit(
                self.symbol,
                current_price
            )

            if action == 'STOP_LOSS':
                logger.warning(f"⚠️ Stop-Loss 도달! 포지션 청산")
                self.close_position(current_price, reason="Stop-Loss")

            elif action == 'TAKE_PROFIT':
                logger.info(f"✅ Take-Profit 도달! 포지션 청산")
                self.close_position(current_price, reason="Take-Profit")

        except Exception as e:
            logger.error(f"SL/TP 확인 실패: {e}")

    def open_position(self, signal, current_price: float):
        """포지션 진입"""
        try:
            # 포지션 크기 계산
            position_info = self.risk_manager.calculate_position_size(
                symbol=self.symbol,
                price=current_price,
                confidence=signal.confidence,
                win_rate=0.55,
                avg_win=0.02,
                avg_loss=0.01
            )

            if position_info['position_size_usdt'] == 0:
                logger.warning("포지션 크기가 0입니다. 진입 취소")
                return False

            # 주문 실행
            if self.paper_trading:
                # 페이퍼 트레이딩 (모의)
                logger.info(f"📝 [페이퍼] {signal.signal.name} 포지션 진입")
                logger.info(f"   가격: {current_price:.2f}")
                logger.info(f"   수량: {position_info['quantity']:.6f}")
                logger.info(f"   크기: {position_info['position_size_usdt']:.2f} USDT")
                logger.info(f"   Stop-Loss: {position_info['stop_loss']:.2f}")
                logger.info(f"   Take-Profit: {position_info['take_profit']:.2f}")

                # 리스크 관리자에 포지션 추가
                self.risk_manager.add_position(
                    symbol=self.symbol,
                    side='BUY' if signal.signal == Signal.BUY else 'SELL',
                    entry_price=current_price,
                    quantity=position_info['quantity'],
                    stop_loss=position_info['stop_loss'],
                    take_profit=position_info['take_profit']
                )

                self.current_position = {
                    'signal': signal.signal,
                    'entry_price': current_price,
                    'quantity': position_info['quantity'],
                    'entry_time': datetime.now()
                }

                return True
            else:
                # 실제 주문 (Phase 5에서 구현 예정)
                logger.warning("실제 거래는 Phase 5에서 구현됩니다")
                return False

        except Exception as e:
            logger.error(f"포지션 진입 실패: {e}")
            return False

    def close_position(self, current_price: float, reason: str = "신호"):
        """포지션 청산"""
        if self.current_position is None:
            return False

        try:
            # 손익 계산
            pnl = self.risk_manager.close_position(self.symbol, current_price)

            if pnl is not None:
                logger.info(f"💰 포지션 청산 완료")
                logger.info(f"   진입: {self.current_position['entry_price']:.2f}")
                logger.info(f"   청산: {current_price:.2f}")
                logger.info(f"   손익: {pnl:.2f} USDT")
                logger.info(f"   이유: {reason}")

                # 성과 추적에 기록
                # 현재 활성 전략 이름 가져오기 (간단히 마지막 선택된 전략으로 가정)
                active_strategy = getattr(self, 'last_selected_strategy', 'Unknown')

                self.performance_tracker.record_trade(
                    strategy_name=active_strategy,
                    timestamp=datetime.now(),
                    side='SELL',
                    price=current_price,
                    quantity=self.current_position['quantity'],
                    pnl=pnl
                )

                self.current_position = None
                return True

        except Exception as e:
            logger.error(f"포지션 청산 실패: {e}")
            return False

    def execute_trading_cycle(self):
        """1회 거래 사이클 실행"""
        logger.info("=" * 70)
        logger.info(f"거래 사이클 #{self.cycle_count + 1} 시작")
        logger.info("=" * 70)

        try:
            # 1. 데이터 수집
            logger.info("[ 1/7 ] 최신 데이터 수집 중...")
            df = self.get_latest_data(days=14)

            if df is None or len(df) < 50:
                logger.error("데이터 부족. 사이클 종료")
                return False

            current_price = df['close'].iloc[-1]
            logger.info(f"✅ 현재 가격: {current_price:.2f}")

            # 2. Stop-Loss / Take-Profit 확인 (포지션이 있을 경우)
            logger.info("[ 2/7 ] Stop-Loss / Take-Profit 확인...")
            self.check_stop_loss_take_profit()

            # 3. 시장 상태 감지
            logger.info("[ 3/7 ] 시장 상태 감지...")
            regime_info = self.regime_detector.detect_regime(df)
            logger.info(f"✅ 시장 상태: {regime_info.regime.value} (신뢰도: {regime_info.confidence:.2%})")

            # 4. 전략 리밸런싱 확인
            logger.info("[ 4/7 ] 전략 리밸런싱 확인...")
            if self.strategy_selector.should_rebalance():
                self.strategy_selector.rebalance(df)
                logger.info("✅ 전략 리밸런싱 완료")
            else:
                logger.info("리밸런싱 불필요")

            # 5. 앙상블 신호 생성
            logger.info("[ 5/7 ] 거래 신호 생성...")
            ensemble_signal = self.strategy_selector.generate_ensemble_signal(df)

            logger.info(f"✅ 신호: {ensemble_signal.signal.name}")
            logger.info(f"   신뢰도: {ensemble_signal.confidence:.2%}")
            logger.info(f"   이유: {ensemble_signal.reason}")

            # 전략별 신호 상세
            if 'strategy_signals' in ensemble_signal.metadata:
                logger.info("   전략별 신호:")
                for name, details in ensemble_signal.metadata['strategy_signals'].items():
                    logger.info(f"     - {name}: {details['signal']} (가중치: {details['weight']:.2%})")

            # 마지막 선택 전략 저장
            weights = ensemble_signal.metadata.get('weights', {})
            self.last_selected_strategy = max(weights, key=weights.get) if weights else 'Ensemble'

            # 6. 거래 실행
            logger.info("[ 6/7 ] 거래 실행 판단...")

            # 개선된 신뢰도 임계값 (조정 가능)
            ENTRY_CONFIDENCE_THRESHOLD = 0.15  # 15% (앙상블 신호는 낮게 나올 수 있음)
            EXIT_CONFIDENCE_THRESHOLD = 0.20   # 20% (청산은 좀 더 신중하게)

            if self.current_position is None:
                # 포지션 없음 → 진입 신호 확인
                if ensemble_signal.signal == Signal.BUY and ensemble_signal.confidence > ENTRY_CONFIDENCE_THRESHOLD:
                    logger.info(f"🚀 매수 신호 발생! 포지션 진입 (신뢰도: {ensemble_signal.confidence:.2%} > {ENTRY_CONFIDENCE_THRESHOLD:.2%})")
                    self.open_position(ensemble_signal, current_price)
                elif ensemble_signal.signal == Signal.SELL and ensemble_signal.confidence > ENTRY_CONFIDENCE_THRESHOLD:
                    # 공매도는 일단 제외 (현물 거래 기준)
                    logger.info("⚠️ 매도 신호이지만 포지션 없음 (공매도 미지원)")
                else:
                    logger.info(f"신호 없음 또는 신뢰도 부족 (현재: {ensemble_signal.confidence:.2%}, 필요: {ENTRY_CONFIDENCE_THRESHOLD:.2%}). 대기")
            else:
                # 포지션 있음 → 청산 신호 확인
                position_side = self.current_position['signal']

                if position_side == Signal.BUY and ensemble_signal.signal == Signal.SELL:
                    if ensemble_signal.confidence > EXIT_CONFIDENCE_THRESHOLD:
                        logger.info(f"📉 매도 신호 발생! 포지션 청산 (신뢰도: {ensemble_signal.confidence:.2%} > {EXIT_CONFIDENCE_THRESHOLD:.2%})")
                        self.close_position(current_price, reason="매도 신호")
                    else:
                        logger.info(f"매도 신호이지만 신뢰도 부족 (현재: {ensemble_signal.confidence:.2%}, 필요: {EXIT_CONFIDENCE_THRESHOLD:.2%}). 유지")
                else:
                    logger.info("포지션 유지")

            # 7. 상태 요약
            logger.info("[ 7/7 ] 상태 요약")
            status = self.risk_manager.get_status()
            logger.info(f"✅ 현재 잔고: {status['current_balance']:.2f} USDT")
            logger.info(f"   총 손익: {status['total_pnl']:.2f} USDT ({status['total_pnl_pct']:.2%})")
            logger.info(f"   일일 손익: {status['daily_pnl']:.2f} USDT ({status['daily_pnl_pct']:.2%})")
            logger.info(f"   포지션 수: {status['num_positions']}")

            # 성과 요약
            logger.info("")
            logger.info(self.performance_tracker.get_summary())

            self.cycle_count += 1
            self.last_check_time = datetime.now()

            logger.info("=" * 70)
            logger.info(f"거래 사이클 #{self.cycle_count} 완료")
            logger.info("=" * 70)
            logger.info("")

            return True

        except Exception as e:
            logger.error(f"거래 사이클 실행 실패: {e}", exc_info=True)
            return False

    def run_continuous(self, interval_minutes: int = 60, max_cycles: Optional[int] = None):
        """
        연속 실행 모드

        Args:
            interval_minutes: 사이클 간격 (분)
            max_cycles: 최대 사이클 수 (None이면 무한)
        """
        self.is_running = True
        logger.info("=" * 70)
        logger.info("자동 거래 봇 시작 (연속 모드)")
        logger.info("=" * 70)
        logger.info(f"사이클 간격: {interval_minutes}분")
        logger.info(f"최대 사이클: {max_cycles if max_cycles else '무제한'}")
        logger.info("")

        try:
            while self.is_running:
                # 거래 사이클 실행
                success = self.execute_trading_cycle()

                if not success:
                    logger.error("사이클 실패. 재시도 대기 중...")

                # 최대 사이클 확인
                if max_cycles and self.cycle_count >= max_cycles:
                    logger.info(f"최대 사이클 ({max_cycles}) 도달. 종료")
                    break

                # 대기
                if self.is_running:
                    logger.info(f"다음 사이클까지 {interval_minutes}분 대기...")
                    time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("사용자에 의해 중단됨")
        except Exception as e:
            logger.error(f"치명적 오류 발생: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        """봇 정지"""
        self.is_running = False
        logger.info("=" * 70)
        logger.info("자동 거래 봇 종료")
        logger.info("=" * 70)

        # 최종 리포트
        status = self.risk_manager.get_status()
        logger.info(f"최종 잔고: {status['current_balance']:.2f} USDT")
        logger.info(f"총 손익: {status['total_pnl']:.2f} USDT ({status['total_pnl_pct']:.2%})")
        logger.info(f"총 사이클: {self.cycle_count}")

        # 전략 성과
        logger.info("")
        logger.info(self.performance_tracker.get_summary())

    def get_status_report(self) -> Dict:
        """현재 상태 리포트"""
        risk_status = self.risk_manager.get_status()

        return {
            'bot': {
                'is_running': self.is_running,
                'cycle_count': self.cycle_count,
                'last_check': self.last_check_time,
                'current_position': self.current_position
            },
            'risk': risk_status,
            'strategies': {
                name: {
                    'metrics': self.performance_tracker.get_metrics(name)
                }
                for name in self.strategies.keys()
            }
        }
