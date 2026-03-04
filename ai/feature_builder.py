"""TF별 특화 피처 빌더 — EWMA 표준화 + 직교 피처 + Cross-TF 감쇠 연결"""
import numpy as np
import pandas as pd
from typing import List


# ── EWMA Z-score 설정 ─────────────────────────────────────────
# span=60: 10s→10분, 1m→1시간, 15m→15시간 effective window
# 레짐 급변 시 rolling보다 빠르게 적응 (지수 감쇠)
_EWMA_SPAN = 60
_EWMA_MIN_PERIODS = 20

# ── 피처 이름 상수 ────────────────────────────────────────────
CORE_FEATURES = [
    'return_n',          # pct_change → EWMA Z-score
    'atr_ratio',         # ATR(14)/close → percentile rank [-1,1]
    'bb_position',       # BB 밴드 내 가격 위치 [-0.5, 0.5] (bb_width와 직교)
    'volume_delta',      # (volume - EMA) / EWMA std
    'volatility_change', # log(vol_short/vol_prev) → EWMA Z-score
    'momentum_slope',    # polyfit slope → EWMA Z-score
]

CANDLE_FEATURES = [
    'di_difference',     # (DI+ − DI−) / 50 → 방향성 지표 [-2, 2]
    'taker_buy_ratio',   # taker_buy_base / volume → 매수 공격성 EWMA Z-score
    'trade_intensity',   # 체결 건수 → EWMA Z-score
    'vwap_deviation',    # (close − VWAP) / ATR → 기관 기준선 괴리 [-3, 3]
]

HTF_FEATURES = [
    'htf_trend_dir',     # EMA(50) 기울기 부호 {-1, 0, 1}
    'vol_cycle_pos',     # 변동성 50-bar 내 순위 [0, 1]
    'regime_prob',       # log(vol_short / vol_long), clip[-3,3]
    'prev_hl_structure', # 50-bar range 내 가격 위치 [0, 1]
]

CROSS_TF_FEATURE = [
    'htf_bias',          # 15m 방향 (EMA 감쇠, [-0.5, 0.5])
]

TIER_MAP = {
    '1m':  CORE_FEATURES + CANDLE_FEATURES + CROSS_TF_FEATURE,   # 11
    '3m':  CORE_FEATURES + CANDLE_FEATURES + CROSS_TF_FEATURE,   # 11
    '5m':  CORE_FEATURES + CANDLE_FEATURES + CROSS_TF_FEATURE,   # 11
    '15m': CORE_FEATURES + HTF_FEATURES + CROSS_TF_FEATURE,      # 11
}

RETURN_PERIOD = {
    '1m': 1, '3m': 1, '5m': 1, '15m': 1,
}

# ── TF별 rolling 윈도우 ─────────────────────────────────────
# 원칙: 피처 윈도우 < 예측 horizon × 10 (인과관계 유지)
# 10s=30초 예측 → 최대 ~18bar(3분), 1m=3분 예측 → 최대 ~30bar
# 15m=45분 예측 → 200bar도 OK (12.5시간 < 45분×17)
_TF_WINDOWS = {
    '1m':  {'rank': 60,  'bb': 20, 'vol_std': 20, 'slope': 10},
    '3m':  {'rank': 80,  'bb': 20, 'vol_std': 20, 'slope': 10},
    '5m':  {'rank': 100, 'bb': 20, 'vol_std': 20, 'slope': 10},
    '15m': {'rank': 200, 'bb': 20, 'vol_std': 20, 'slope': 10},
}


class TFFeatureBuilder:
    """타임프레임별 피처를 생성하는 빌더

    설계 원칙:
    - EWMA Z-score: 레짐 급변에 빠르게 적응 (rolling 대비)
    - 직교 피처: atr_ratio(변동성 수준) vs bb_position(밴드 내 위치) — 중복 제거
    - Cross-TF 감쇠: htf_bias가 학습을 지배하지 않도록 EMA 감쇠 + 스케일 축소
    """

    def __init__(self):
        self._obi_cache: float = 0.0
        self._obi_raw: float = 0.0      # EMA 평활 전 raw
        # Cross-TF: 15m 방향 (EMA 감쇠 적용)
        self._htf_direction_cache: float = 0.0
        self._htf_decay_alpha: float = 0.05  # 호출마다 5% 감쇠 → 0에 수렴

    # ── public API ────────────────────────────────────────────

    def get_feature_names(self, timeframe: str) -> List[str]:
        return list(TIER_MAP.get(timeframe, CORE_FEATURES + CANDLE_FEATURES + CROSS_TF_FEATURE))

    def get_input_dim(self, timeframe: str) -> int:
        return len(self.get_feature_names(timeframe))

    def update_obi(self, obi_value: float):
        """OBI 업데이트 — EMA 평활 (alpha=0.3)으로 노이즈/스푸핑 완화"""
        alpha = 0.3
        self._obi_raw = obi_value
        self._obi_cache = alpha * obi_value + (1 - alpha) * self._obi_cache

    def prepare_features(self, df: pd.DataFrame, timeframe: str) -> np.ndarray:
        """OHLCV DataFrame → (rows, n_features) ndarray"""
        df = df.copy()
        close = df['close']
        wins = _TF_WINDOWS.get(timeframe, _TF_WINDOWS['1m'])

        period = RETURN_PERIOD.get(timeframe, 1)
        self._compute_core(df, close, period, wins)

        if timeframe == '15m':
            self._compute_htf_features(df, close)
        else:
            self._compute_candle_features(df, close, wins)

        self._compute_cross_tf(df, timeframe)

        feature_names = self.get_feature_names(timeframe)
        features = df[feature_names].values

        features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)
        features = np.clip(features, -5.0, 5.0)
        return features

    # ── EWMA Z-score 유틸 ────────────────────────────────────

    @staticmethod
    def _ewma_zscore(series: pd.Series,
                     span: int = _EWMA_SPAN,
                     min_periods: int = _EWMA_MIN_PERIODS) -> pd.Series:
        """EWMA 기반 Z-score — 레짐 급변 시 rolling보다 빠르게 적응

        EWM std는 최근 데이터에 지수 가중 → 변동성 폭발 직후, 세션 전환,
        뉴스 직후에도 분모가 빠르게 갱신됨.
        """
        ewm = series.ewm(span=span, min_periods=min_periods)
        mean = ewm.mean()
        std = ewm.std().replace(0, 1)
        return (series - mean) / std

    # ── 코어 피처 (6개) ──────────────────────────────────────

    def _compute_core(self, df: pd.DataFrame, close: pd.Series, period: int, wins: dict):
        # 1) return_n — pct_change → EWMA Z-score
        raw_ret = close.pct_change(period) * 100
        df['return_n'] = self._ewma_zscore(raw_ret)

        # 2) atr_ratio — ATR(14)/close → percentile rank [-1, 1]
        #    TF별 rank 윈도우: 10s=50, 1m=60, 15m=200
        high_low = df['high'] - df['low']
        high_close = (df['high'] - close.shift()).abs()
        low_close = (df['low'] - close.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=1).mean()
        atr_ratio_raw = atr / close
        rank_win = wins['rank']
        df['atr_ratio'] = atr_ratio_raw.rolling(rank_win, min_periods=min(20, rank_win)).rank(pct=True) * 2 - 1

        # 3) bb_position — BB 밴드 내 가격 위치 [-0.5, 0.5]
        bb_win = wins['bb']
        bb_middle = close.rolling(bb_win, min_periods=1).mean()
        bb_std_val = close.rolling(bb_win, min_periods=1).std().replace(0, 1e-10)
        bb_upper = bb_middle + 2 * bb_std_val
        bb_lower = bb_middle - 2 * bb_std_val
        bb_range = (bb_upper - bb_lower).replace(0, 1e-10)
        df['bb_position'] = (close - bb_lower) / bb_range - 0.5

        # 4) volume_delta — EWMA Z-score
        vol = df['volume']
        df['volume_delta'] = self._ewma_zscore(vol)

        # 5) volatility_change — log(vol_short / vol_prev) → EWMA Z-score
        vol_win = wins['vol_std']
        vol_s = raw_ret.rolling(vol_win, min_periods=max(5, vol_win // 2)).std()
        vol_prev = vol_s.shift(5).replace(0, 1e-10)
        vc_log = np.log((vol_s / vol_prev).clip(lower=1e-10))
        df['volatility_change'] = self._ewma_zscore(vc_log).clip(-3, 3)

        # 6) momentum_slope — polyfit slope → EWMA Z-score
        slope_win = wins['slope']
        ret_series = raw_ret.fillna(0)
        slopes = self._rolling_slope(ret_series, window=slope_win)
        slopes_s = pd.Series(slopes, index=df.index)
        df['momentum_slope'] = self._ewma_zscore(slopes_s)

    # ── 1m~5m 캔들 전용 (4개) ────────────────────────────────

    def _compute_candle_features(self, df: pd.DataFrame, close: pd.Series, wins: dict):
        # 1) di_difference — (DI+ − DI−) / 50 → 방향성 지표 [-2, 2]
        high = df['high']
        low = df['low']
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_14 = tr.rolling(14, min_periods=1).mean().replace(0, 1e-10)

        plus_di = 100 * plus_dm.rolling(14, min_periods=1).mean() / atr_14
        minus_di = 100 * minus_dm.rolling(14, min_periods=1).mean() / atr_14
        df['di_difference'] = ((plus_di - minus_di) / 50.0).clip(-2, 2)

        # 2) taker_buy_ratio — 매수/매도 공격성 비율 → EWMA Z-score
        if 'taker_buy_base' in df.columns:
            vol_safe = df['volume'].replace(0, 1e-10)
            raw_ratio = df['taker_buy_base'].astype(float) / vol_safe
        else:
            raw_ratio = pd.Series(0.5, index=df.index)
        df['taker_buy_ratio'] = self._ewma_zscore(raw_ratio)

        # 3) trade_intensity — 체결 건수 이상 감지 → EWMA Z-score
        if 'trades' in df.columns:
            df['trade_intensity'] = self._ewma_zscore(df['trades'].astype(float))
        else:
            df['trade_intensity'] = 0.0

        # 4) vwap_deviation — (close − rolling VWAP) / ATR → [-3, 3]
        typical_price = (high + low + close) / 3
        vwap_win = wins['bb']  # 20 bars
        cum_tp_vol = (typical_price * df['volume']).rolling(vwap_win, min_periods=1).sum()
        cum_vol = df['volume'].rolling(vwap_win, min_periods=1).sum().replace(0, 1e-10)
        vwap = cum_tp_vol / cum_vol
        df['vwap_deviation'] = ((close - vwap) / atr_14).clip(-3, 3)

    # ── 15m HTF 전용 (4개) ───────────────────────────────────

    def _compute_htf_features(self, df: pd.DataFrame, close: pd.Series):
        # 1) htf_trend_dir — EMA(50) 기울기 부호 {-1, 0, 1}
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema_slope = ema50.diff(5)
        htf_dir = np.sign(ema_slope).fillna(0)
        df['htf_trend_dir'] = htf_dir

        # 캐시 갱신 (하위 TF 전달용, EMA 감쇠는 _compute_cross_tf에서)
        last_valid = htf_dir.dropna()
        if len(last_valid) > 0:
            self._htf_direction_cache = float(last_valid.iloc[-1])

        # 2) vol_cycle_pos — 변동성의 50-bar 순위 [0, 1]
        ret_vol = close.pct_change().rolling(20, min_periods=5).std()
        df['vol_cycle_pos'] = ret_vol.rolling(50, min_periods=10).rank(pct=True)

        # 3) regime_prob — log(vol_short / vol_long), clip[-3,3]
        vol_short = close.pct_change().rolling(10, min_periods=5).std()
        vol_long = close.pct_change().rolling(50, min_periods=10).std().replace(0, 1e-10)
        df['regime_prob'] = np.log((vol_short / vol_long).clip(lower=1e-6)).clip(-3, 3)

        # 4) prev_hl_structure — 50-bar range 내 가격 위치 [0, 1]
        high_50 = df['high'].rolling(50, min_periods=1).max()
        low_50 = df['low'].rolling(50, min_periods=1).min()
        rng = (high_50 - low_50).replace(0, 1e-10)
        df['prev_hl_structure'] = (close - low_50) / rng

    # ── Cross-TF 연결 (1개) ──────────────────────────────────

    def _compute_cross_tf(self, df: pd.DataFrame, timeframe: str):
        """15m HTF 방향 → 하위 TF 전달 (감쇠 + 스케일 축소)

        - EMA 감쇠: 15m 갱신 없이 하위 TF만 호출되면 캐시가 0에 수렴
        - 스케일 [-0.5, 0.5]: 다른 피처 대비 magnitude 절반 → dominance 방지
        """
        if timeframe == '15m':
            # 15m 자체: htf_trend_dir 자기 참조, 스케일 축소
            df['htf_bias'] = df.get('htf_trend_dir', 0.0) * 0.5
        else:
            # 하위 TF: 캐시값 사용 + 감쇠
            bias = self._htf_direction_cache * 0.5
            df['htf_bias'] = bias
            # 호출마다 감쇠: 15m 갱신 없이 계속 호출되면 서서히 0으로
            self._htf_direction_cache *= (1 - self._htf_decay_alpha)

    # ── 유틸 ─────────────────────────────────────────────────

    @staticmethod
    def _rolling_slope(series: pd.Series, window: int = 10) -> np.ndarray:
        """최근 window개 값에 대한 선형회귀 기울기를 rolling 으로 계산"""
        values = series.values.astype(float)
        n = len(values)
        slopes = np.zeros(n)
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        if x_var == 0:
            return slopes
        for i in range(window - 1, n):
            y = values[i - window + 1: i + 1]
            y_mean = y.mean()
            slopes[i] = ((x - x_mean) * (y - y_mean)).sum() / x_var
        return slopes

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index 계산"""
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=1).mean().replace(0, 1e-10)

        plus_di = 100 * plus_dm.rolling(period, min_periods=1).mean() / atr
        minus_di = 100 * minus_dm.rolling(period, min_periods=1).mean() / atr

        dx_denom = (plus_di + minus_di).replace(0, 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / dx_denom
        adx = dx.rolling(period, min_periods=1).mean()
        return adx

    @staticmethod
    def compute_raw_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Raw ATR(period) — 라벨링용 (percentile rank 아님).
        Returns: ATR in absolute price units.
        """
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()
