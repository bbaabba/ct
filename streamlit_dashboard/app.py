"""선물 트레이딩 봇 실시간 대시보드"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

from streamlit_dashboard.state_reader import BotStateReader

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI 선물 트레이딩 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 자동 새로고침 (3초) ──────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=3000, key="dashboard_refresh")
except ImportError:
    # streamlit-autorefresh 미설치 시 meta 태그 폴백
    st.markdown(
        '<meta http-equiv="refresh" content="3">',
        unsafe_allow_html=True
    )

# ── 스타일 ───────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric > div { padding: 8px 0; }
    .trade-profit { color: #00c853; font-weight: bold; }
    .trade-loss { color: #ff1744; font-weight: bold; }
    div[data-testid="stMetricDelta"] > div { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── 데이터 로드 ──────────────────────────────────────────────
STATE_DIR = os.path.join(PROJECT_ROOT, "bot_state")
reader = BotStateReader(state_dir=STATE_DIR)

status = reader.read_status()
trades = reader.read_trade_history()
candles_df = reader.read_candles()
missed_opps = reader.read_missed_opportunities()
data_age = reader.get_data_age_seconds()


# ══════════════════════════════════════════════════════════════
# 헤더
# ══════════════════════════════════════════════════════════════
st.title("📊 AI 선물 트레이딩 대시보드")

# 데이터 신선도 표시
if data_age is not None:
    if data_age > 30:
        st.error(f"⚠️ 봇 연결 끊김 — 마지막 업데이트: {data_age:.0f}초 전")
    elif data_age > 10:
        st.warning(f"⏳ 데이터 지연 — {data_age:.0f}초 전 업데이트")
    else:
        st.caption(f"✅ 실시간 | 마지막 업데이트: {data_age:.1f}초 전")
else:
    st.info("🔌 봇이 실행 중이 아니거나 상태 파일이 아직 생성되지 않았습니다.")
    st.caption("봇을 먼저 실행해주세요: `python main.py` 또는 `python run_with_dashboard.py`")
    st.stop()

if status is None:
    st.error("봇 상태를 읽을 수 없습니다.")
    st.stop()


# ══════════════════════════════════════════════════════════════
# 페이지 네비게이션
# ══════════════════════════════════════════════════════════════
_page = st.radio(
    "페이지", ["📈 트레이딩", "🧠 AI 모델 통계"],
    horizontal=True, label_visibility="collapsed"
)


# ══════════════════════════════════════════════════════════════
# AI 모델 통계 페이지
# ══════════════════════════════════════════════════════════════
if _page == "🧠 AI 모델 통계":
    _model_stats = status.get('model_stats', {})
    _tf_stats = _model_stats.get('timeframes', {})
    _tick_stats = _model_stats.get('tick')
    _unified_ms = _model_stats.get('unified')
    _cycle_ctx_ms = status.get('cycle_ctx', {})
    _is_unified_ms = _cycle_ctx_ms.get('use_unified_model', False) if _cycle_ctx_ms else False

    # ══════════════════════════════════════════════════════════
    # 통합 모델 통계
    # ══════════════════════════════════════════════════════════
    if _is_unified_ms:
        st.subheader("🧬 통합 모델 (Multi-Scale TCN + MoE)")

        if _unified_ms:
            # Shadow Mode 상태
            _sh_state = _unified_ms.get('shadow_state', '-')
            _sh_map = {'SHADOW': '⏳ 가상 거래', 'LIVE': '🟢 실전 활성', 'DEMOTED': '🔴 강등'}
            _sh_c1, _sh_c2, _sh_c3 = st.columns(3)
            _sh_c1.metric("Shadow 상태", _sh_map.get(_sh_state, _sh_state))
            _sh_c2.metric("해소된 예측", f"{_unified_ms.get('shadow_resolved', 0)}건")
            _sh_c3.metric("대기 중 예측", f"{_unified_ms.get('shadow_pending', 0)}건")

            _sh_c4, _sh_c5, _sh_c6 = st.columns(3)
            _sh_acc = _unified_ms.get('shadow_accuracy', 0)
            _sh_pf = _unified_ms.get('shadow_profit_factor', 0)
            _sh_c4.metric("Shadow 정확도", f"{_sh_acc:.1%}")
            _sh_c5.metric("Profit Factor", f"{_sh_pf:.2f}")
            _sh_prom = _unified_ms.get('shadow_promotions', 0)
            _sh_dem = _unified_ms.get('shadow_demotions', 0)
            _sh_c6.metric("승격/강등", f"{_sh_prom}/{_sh_dem}")

            st.divider()

            # MoE 라우터 상태
            st.subheader("🎯 MoE 전문가 라우팅")
            _router_sharp = _unified_ms.get('router_sharpness', 0)
            _ew = _unified_ms.get('expert_weights', [])

            _moe_c1, _moe_c2 = st.columns(2)
            if _router_sharp > 0.5:
                _sharp_label = "🟢 선택적"
            elif _router_sharp > 0.2:
                _sharp_label = "🟡 보통"
            else:
                _sharp_label = "🔴 균일"
            _moe_c1.metric("Router Sharpness", f"{_router_sharp:.3f}", delta=_sharp_label)

            _moe_comb = _unified_ms.get('moe_combined')
            _moe_mult = _unified_ms.get('moe_size_mult')
            if _moe_comb is not None:
                _moe_c2.metric("MoE 확신도", f"{_moe_comb:.3f}",
                               delta=f"사이즈 {_moe_mult:.2f}x" if _moe_mult else None)

            # 전문가 가중치 바 차트
            if _ew:
                _expert_names = ['Macro (15m)', 'Mid (3m)', 'Micro (10s)', 'Cross-TF']
                _ew_rows = []
                for _ei, _w in enumerate(_ew):
                    _en = _expert_names[_ei] if _ei < len(_expert_names) else f"Expert {_ei}"
                    _ew_rows.append({'전문가': _en, '가중치': _w})
                _ew_df = pd.DataFrame(_ew_rows)
                st.bar_chart(_ew_df.set_index('전문가'), height=200)
        else:
            st.info("통합 모델 통계 없음 — 아직 학습이 시작되지 않았습니다.")

    # ══════════════════════════════════════════════════════════
    # 레거시 MTF 모델 통계
    # ══════════════════════════════════════════════════════════
    else:
        st.subheader("📊 시간대별 모델 상태")

        if _tf_stats:
            _tf_order = ['15m', '5m', '3m', '1m']
            _sorted_tfs = [tf for tf in _tf_order if tf in _tf_stats]
            _sorted_tfs += [tf for tf in _tf_stats if tf not in _tf_order]

            _summary_rows = []
            for _tf in _sorted_tfs:
                _s = _tf_stats[_tf]
                _dir_acc = _s.get('direction_accuracy', 0)
                _bal_acc = _s.get('balanced_accuracy', 0)
                _tc = _s.get('training_count', 0)
                _pc = _s.get('prediction_count', 0)
                _exp = _s.get('expectancy', 0)
                _shrp = _s.get('sharpe', 0)
                _vl = _s.get('best_val_loss')
                _deg = _s.get('consecutive_degrades', 0)
                _pre = '✅' if _s.get('is_pretrained') else '❌'

                _summary_rows.append({
                    '시간대': _tf,
                    '사전학습': _pre,
                    '학습횟수': _tc,
                    '예측수': _pc,
                    '방향정확도': f"{_dir_acc:.1%}",
                    'Balanced Acc': f"{_bal_acc:.1%}",
                    'Expectancy': f"{_exp:+.3f}%",
                    'Sharpe': f"{_shrp:.2f}",
                    'Val Loss': f"{_vl:.4f}" if _vl and _vl < 100 else '-',
                    '연속악화': f"{'⚠️ ' if _deg >= 2 else ''}{_deg}",
                })

            st.dataframe(pd.DataFrame(_summary_rows), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("🔍 시간대별 상세")

            _cols = st.columns(len(_sorted_tfs))
            for _i, _tf in enumerate(_sorted_tfs):
                _s = _tf_stats[_tf]
                with _cols[_i]:
                    _role_map = {'15m': 'Step1 추세', '5m': 'Step2 모멘텀', '3m': 'Step2 모멘텀', '1m': 'Step3 미시필터'}
                    st.markdown(f"### {_tf} ({_role_map.get(_tf, '')})")

                    _tc = _s.get('training_count', 0)
                    _pre = _s.get('is_pretrained', False)
                    st.metric("학습 횟수", f"{_tc}회", delta="사전학습됨" if _pre else "새 모델")

                    _bal_acc = _s.get('balanced_accuracy', 0)
                    _dir_acc = _s.get('direction_accuracy', 0)
                    _maturity = max(0, min(1, (_bal_acc - 0.40) * 5))
                    if _maturity < 0.1:
                        _mat_label = "🔴 차단"
                    elif _maturity < 0.5:
                        _mat_label = "🟡 감쇠"
                    else:
                        _mat_label = "🟢 정상"
                    st.metric("Balanced Acc", f"{_bal_acc:.1%}", delta=_mat_label)
                    st.caption(f"Simple Acc: {_dir_acc:.1%} | 성숙도: {_maturity:.0%}")

                    _exp = _s.get('expectancy', 0)
                    _shrp = _s.get('sharpe', 0)
                    st.metric("Expectancy", f"{_exp:+.3f}%")
                    st.metric("Sharpe", f"{_shrp:.2f}")

                    st.markdown("---")
                    st.caption(f"학습 버퍼: {_s.get('buffer_size', 0)}개")
                    st.caption(f"리플레이: {_s.get('replay_size', 0)}개")
                    st.caption(f"링버퍼 대기: {_s.get('ring_buffer_pending', 0)}개")
                    st.caption(f"라벨링 완료: {_s.get('ring_buffer_labeled', 0)}개")

                    _vl = _s.get('best_val_loss')
                    _deg = _s.get('consecutive_degrades', 0)
                    if _vl and _vl < 100:
                        st.caption(f"Best Val Loss: {_vl:.4f}")
                    if _deg >= 2:
                        st.warning(f"연속 악화: {_deg}회")

                    _lt = _s.get('last_training')
                    if _lt:
                        try:
                            _lt_dt = datetime.fromisoformat(_lt)
                            _elapsed = (datetime.now() - _lt_dt).total_seconds() / 60
                            st.caption(f"마지막 학습: {_elapsed:.0f}분 전")
                        except (ValueError, TypeError):
                            pass
        else:
            st.info("모델 통계 데이터 없음 — 봇이 실행 중이 아니거나 아직 학습이 시작되지 않았습니다.")

        # ── 10s 틱 모델 (레거시 전용) ──
        st.divider()
        st.subheader("⚡ 10s 틱 모델 (Step4 실행게이트)")

        if _tick_stats:
            _tc1, _tc2, _tc3, _tc4 = st.columns(4)
            _tc1.metric("학습 횟수", f"{_tick_stats.get('training_count', 0)}회",
                         delta="사전학습됨" if _tick_stats.get('is_pretrained') else "새 모델")
            _tc2.metric("예측 수", f"{_tick_stats.get('prediction_count', 0)}건")
            _tc3.metric("학습 버퍼", f"{_tick_stats.get('buffer_size', 0)}개")
            _tc4.metric("라벨링 완료", f"{_tick_stats.get('ring_buffer_labeled', 0)}개")

            _lt = _tick_stats.get('last_training')
            if _lt:
                try:
                    _lt_dt = datetime.fromisoformat(_lt)
                    _elapsed = (datetime.now() - _lt_dt).total_seconds() / 60
                    st.caption(f"마지막 학습: {_elapsed:.0f}분 전 | 링버퍼 대기: {_tick_stats.get('ring_buffer_pending', 0)}개")
                except (ValueError, TypeError):
                    pass
        else:
            st.info("틱 모델 비활성 — 이 모드에서는 별도의 틱 모델이 사용되지 않습니다.")

    # ── PriceLog 상태 (공통) ──
    _pl_size = _model_stats.get('price_log_size', 0)
    if _pl_size:
        st.caption(f"📊 PriceLog: {_pl_size:,}개 데이터포인트 (~{_pl_size * 10 / 3600:.1f}시간)")

    # 모델 페이지에서도 사이드바 표시 후 중단
    _sb_status = status
    with st.sidebar:
        st.header("📡 데이터 상태")
        _ds = _sb_status.get('data_status')
        if _ds:
            st.metric("현재가", f"{_ds.get('current_price', 0):,.2f}")
            st.metric("WebSocket", "🟢 연결됨" if _ds.get('is_connected') else "🔴 끊김")
            st.metric("캔들 수", _ds.get('candle_count', 0))
        st.divider()
        st.header("⚙️ 시스템")
        _mode_label = "🧬 통합 모델" if _is_unified_ms else "🔗 MTF Cascade"
        st.caption(f"모드: {_mode_label}")
        st.caption(f"봇 상태: {_sb_status.get('state', '-')}")
        st.caption(f"심볼: {_sb_status.get('symbol', '-')}")
        if data_age is not None:
            st.caption(f"데이터 경과: {data_age:.1f}초")
    st.stop()


# ══════════════════════════════════════════════════════════════
# 트레이딩 페이지 (기존 코드 — 변경 없음)
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# Row 1: 핵심 메트릭
# ══════════════════════════════════════════════════════════════
st.divider()

bot_state = status.get('state', 'UNKNOWN')
state_map = {
    'RUNNING': ('🟢 실행중', 'normal'),
    'PAUSED': ('🟡 일시정지', 'normal'),
    'STOPPED': ('🔴 정지', 'off'),
    'ERROR': ('🔴 오류', 'off'),
    'IDLE': ('⚪ 대기', 'normal'),
}
state_display, _ = state_map.get(bot_state, (f'⚪ {bot_state}', 'normal'))

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("봇 상태", state_display)

balance = status.get('current_balance', 0)
initial = status.get('initial_balance', balance)
balance_delta = balance - initial if initial else 0
col2.metric("잔고 (USDT)", f"{balance:,.2f}", delta=f"{balance_delta:+,.2f}")

total_pnl = status.get('total_pnl', 0)
col3.metric("총 손익", f"{total_pnl:+,.2f} USDT")

daily_pnl = status.get('daily_pnl', 0)
col4.metric("일일 손익", f"{daily_pnl:+,.2f} USDT")

default_lev = status.get('default_leverage', '-')
current_lev = status.get('current_leverage', '-')
col5.metric("레버리지", f"{current_lev}x", delta=f"기본: {default_lev}x")

symbol = status.get('symbol', '-')
col6.metric("심볼", symbol)


# ══════════════════════════════════════════════════════════════
# Row 2: 현재 포지션
# ══════════════════════════════════════════════════════════════
st.subheader("📈 현재 포지션")

position = status.get('current_position')
if position and position.get('side'):
    pcol1, pcol2, pcol3, pcol4, pcol5, pcol6 = st.columns(6)

    side = position.get('side', '')
    side_display = "🟢 LONG" if side == 'LONG' else "🔴 SHORT"
    pcol1.metric("방향", side_display)

    entry_price = position.get('entry_price', 0)
    pcol2.metric("진입가", f"{entry_price:,.2f}")

    quantity = position.get('quantity', 0)
    pcol3.metric("수량", f"{quantity:.4f}")

    leverage = position.get('leverage', 1)
    pcol4.metric("포지션 레버리지", f"{leverage}x")

    sl = position.get('stop_loss')
    pcol5.metric("손절가", f"{sl:,.2f}" if sl else "-")

    tp = position.get('take_profit')
    pcol6.metric("익절가", f"{tp:,.2f}" if tp else "-")

    # 미실현 손익 (status에 포함된 경우)
    unrealized = status.get('unrealized_pnl')
    current_price = status.get('current_price', 0)
    if unrealized is not None:
        st.metric("미실현 손익", f"{unrealized:+,.2f} USDT (현재가: {current_price:,.2f})")
else:
    st.info("현재 열린 포지션이 없습니다")


# ══════════════════════════════════════════════════════════════
# Row 3: 거래 내역 (차트 위로 이동)
# ══════════════════════════════════════════════════════════════
_th_col1, _th_col2 = st.columns([6, 1])
with _th_col1:
    st.subheader("📋 거래 내역")
with _th_col2:
    if st.button("🗑️ 초기화", key="clear_trades", help="거래 내역을 초기화합니다"):
        _confirm_key = "confirm_clear_trades"
        st.session_state[_confirm_key] = True

if st.session_state.get("confirm_clear_trades"):
    st.warning("정말로 거래 내역을 초기화하시겠습니까? 이 작업은 되돌릴 수 없습니다.")
    _cc1, _cc2, _cc3 = st.columns([1, 1, 4])
    with _cc1:
        if st.button("✅ 확인", key="do_clear"):
            import json as _json
            _th_path = os.path.join(STATE_DIR, "trade_history.json")
            with open(_th_path, 'w', encoding='utf-8') as _f:
                _json.dump({'trades': [], 'total_count': 0, '_written_at': datetime.now().isoformat()}, _f, ensure_ascii=False)
            st.session_state["confirm_clear_trades"] = False
            st.success("거래 내역이 초기화되었습니다.")
            st.rerun()
    with _cc2:
        if st.button("❌ 취소", key="cancel_clear"):
            st.session_state["confirm_clear_trades"] = False
            st.rerun()

selected_trade = None
if trades:
    trades_df = pd.DataFrame(trades)

    # 컬럼 매핑
    column_map = {
        'side': '방향',
        'entry_price': '진입가',
        'exit_price': '청산가',
        'quantity': '수량',
        'leverage': '레버리지',
        'fee': '수수료',
        'pnl': '순손익 (USDT)',
        'pnl_pct': '손익률 (%)',
        'reason': '사유',
        'entry_time': '진입시간',
        'exit_time': '청산시간'
    }

    display_cols = [c for c in column_map.keys() if c in trades_df.columns]
    display_df = trades_df[display_cols].rename(columns=column_map)

    # 최신순 정렬
    display_df = display_df.iloc[::-1].reset_index(drop=True)

    # 숫자 컬럼 포맷팅
    display_df_formatted = display_df.copy()
    _fmt_map = {
        '진입가': lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x,
        '청산가': lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x,
        '수량': lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x,
        '수수료': lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x,
        '순손익 (USDT)': lambda x: f"{x:+,.2f}" if isinstance(x, (int, float)) else x,
        '손익률 (%)': lambda x: f"{x:+,.2f}%" if isinstance(x, (int, float)) else x,
    }
    for col_name, fmt_fn in _fmt_map.items():
        if col_name in display_df_formatted.columns:
            display_df_formatted[col_name] = display_df_formatted[col_name].apply(fmt_fn)

    # 선택 가능한 데이터프레임으로 표시
    st.caption("💡 거래를 클릭하면 차트에 진입/청산 시점이 표시됩니다")

    event = st.dataframe(
        display_df_formatted,
        use_container_width=True,
        height=min(400, 50 + len(display_df_formatted) * 35),
        selection_mode="single-row",
        on_select="rerun",
        key="trades_table"
    )

    # 선택된 행 확인
    if event.selection and len(event.selection.rows) > 0:
        selected_idx = event.selection.rows[0]
        # 역순으로 정렬했으므로 원래 인덱스 계산
        original_idx = len(trades_df) - 1 - selected_idx
        selected_trade = trades_df.iloc[original_idx].to_dict()

        # 선택된 거래 정보 표시
        st.success(f"✅ 선택됨: {selected_trade.get('side')} 포지션 (진입: {selected_trade.get('entry_time')}, 청산: {selected_trade.get('exit_time')})")

        # AI 판단 정보 표시
        entry_ai = selected_trade.get('entry_ai_info', {})
        exit_ai = selected_trade.get('exit_ai_info', {})

        if entry_ai or exit_ai:
            ai_col1, ai_col2 = st.columns(2)

            def _render_ai_panel(label, ai_data):
                """진입/청산 AI 판단 패널 렌더링
                Cascade 흐름 순서: 최종판단 → Cascade 입력(15m→5m/3m→1m→10s) → 필터
                """
                st.markdown(f"**{label}**")
                if not ai_data:
                    st.caption("AI 정보 없음 (이전 버전 거래)")
                    return

                # ── 1. Cascade 최종 판단 (실제 진입 결정) ──
                cyc = ai_data.get('cycle', {})
                if cyc:
                    _c_dir = cyc.get('cascade_dir')
                    _c_conf = cyc.get('cascade_conf')
                    _c_fire = cyc.get('cascade_fire')
                    _final = cyc.get('final_score')

                    st.markdown("**Cascade 최종 판단**")
                    if _c_dir:
                        _dir_icon = '🟢' if _c_dir == 'LONG' else '🔴' if _c_dir == 'SHORT' else '⚪'
                        _fire_icon = '🔥' if _c_fire else '⏳'
                        st.markdown(f"- 방향: {_dir_icon} **{_c_dir}** {_fire_icon}")
                    if _c_conf is not None:
                        st.markdown(f"- 신뢰도: **{_c_conf:.1%}**")
                    if _final is not None:
                        st.markdown(f"- 최종 스코어: **{_final:+.3f}**")

                    # 성숙도
                    _acc = cyc.get('ai_accuracy')
                    _mat = cyc.get('ai_maturity')
                    _pcnt = cyc.get('pred_count', 0)
                    if _mat is not None:
                        _mat_icon = '🔴' if _mat < 0.1 else '🟡' if _mat < 0.5 else '🟢'
                        st.markdown(f"- 성숙도: {_mat_icon} **{_mat:.2f}** (정확도 {_acc:.1%}, 예측 {_pcnt}건)" if _acc is not None else f"- 성숙도: {_mat_icon} **{_mat:.2f}**")

                    # 수수료 EV
                    _fee_ev = cyc.get('fee_ev_ratio')
                    if _fee_ev is not None:
                        st.markdown(f"- Fee EV: **{_fee_ev:.1f}x**")

                # ── 2. Cascade 입력 (각 Step 예측) ──
                _mtf = ai_data.get('mtf_predictions', {})
                _dir_1m = ai_data.get('recommended_direction')
                _conf_1m = ai_data.get('combined_confidence')
                _tick_dir = ai_data.get('tick_direction')
                _tick_timing = ai_data.get('tick_timing')

                _has_inputs = _mtf or _dir_1m or _tick_dir
                if _has_inputs:
                    st.markdown("---")
                    st.markdown("**Cascade 입력**")
                    # Step1: 15m (추세)
                    _p15 = _mtf.get('15m')
                    if _p15:
                        _d = _p15.get('direction', '-')
                        _c = _p15.get('confidence', 0)
                        _icon = '🟢' if _d == 'UP' else '🔴' if _d == 'DOWN' else '⚪'
                        st.markdown(f"- Step1 15m 추세: {_icon} **{_d}** ({_c:.1%})")
                    # Step2: 5m/3m (모멘텀)
                    for _tf in ['5m', '3m']:
                        _p = _mtf.get(_tf)
                        if _p:
                            _d = _p.get('direction', '-')
                            _c = _p.get('confidence', 0)
                            _icon = '🟢' if _d == 'UP' else '🔴' if _d == 'DOWN' else '⚪'
                            st.markdown(f"- Step2 {_tf} 모멘텀: {_icon} **{_d}** ({_c:.1%})")
                    # Step3: 1m (미시필터)
                    if _dir_1m and _dir_1m != '-':
                        st.markdown(f"- Step3 1m 미시: **{_dir_1m}**" + (f" ({_conf_1m:.1%})" if _conf_1m is not None else ""))
                    # Step4: 10s (실행게이트)
                    if _tick_dir and _tick_dir != '-':
                        st.markdown(f"- Step4 10s 실행: **{_tick_dir}**" + (f" (타이밍 {_tick_timing:.1%})" if _tick_timing is not None else ""))

                # ── 3. 컨텍스트 (레짐, 변동성, 필터) ──
                if cyc:
                    _regime = cyc.get('regime')
                    _regime_conf = cyc.get('regime_conf')
                    _vf = cyc.get('vol_factor')
                    _vz = cyc.get('vol_z')
                    _thr = cyc.get('hold_threshold')

                    _has_ctx = _regime or _vf is not None or _thr is not None
                    if _has_ctx:
                        st.markdown("---")
                        st.markdown("**필터 & 컨텍스트**")
                        if _regime:
                            st.markdown(f"- 레짐: **{_regime}**" + (f" ({_regime_conf:.0%})" if _regime_conf is not None else ""))
                        if _thr is not None:
                            st.markdown(f"- 진입 임계값: **{_thr}**")
                        if _vf is not None:
                            st.markdown(f"- 변동성: vol_z={_vz:+.2f} → factor=**{_vf:.2f}**" if _vz is not None else f"- vol_factor: **{_vf:.2f}**")

                    # TA 개별 신호
                    _ta_sigs = cyc.get('ta_signals', {})
                    if _ta_sigs:
                        st.markdown("---")
                        st.markdown("**TA 개별 신호**")
                        for sig_name, sig_val in _ta_sigs.items():
                            if isinstance(sig_val, (int, float)):
                                _bar_len = max(0, min(10, int(sig_val * 10)))
                                _bar = '🟩' * _bar_len + '⬜' * (10 - _bar_len)
                                _icon = '🟢' if sig_val > 0.55 else '🔴' if sig_val < 0.45 else '⚪'
                                st.markdown(f"- {sig_name}: {_bar} **{sig_val:.2f}** {_icon}")

            with ai_col1:
                _render_ai_panel("진입 시 AI 판단", entry_ai)

            with ai_col2:
                _render_ai_panel("청산 시 AI 판단", exit_ai)

    # 손익 요약
    if 'pnl' in trades_df.columns:
        profit_trades = trades_df[trades_df['pnl'] > 0]
        loss_trades = trades_df[trades_df['pnl'] <= 0]
        avg_profit = profit_trades['pnl'].mean() if len(profit_trades) > 0 else 0
        avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0

        # 거래당 평균수익률
        if 'pnl_pct' in trades_df.columns:
            avg_return_pct = trades_df['pnl_pct'].mean()
        else:
            avg_return_pct = trades_df['pnl'].mean()

        # 총 수수료 계산
        total_fee = trades_df['fee'].sum() if 'fee' in trades_df.columns else 0

        sumcol1, sumcol2, sumcol3, sumcol4, sumcol5 = st.columns(5)
        sumcol1.metric("거래당 평균수익률", f"{avg_return_pct:+,.2f}%")
        sumcol2.metric("평균 수익", f"{avg_profit:+,.2f} USDT")
        sumcol3.metric("평균 손실", f"{avg_loss:+,.2f} USDT")
        profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        sumcol4.metric("Profit Factor", f"{profit_factor:.2f}")
        sumcol5.metric("총 수수료", f"{total_fee:,.2f} USDT")
else:
    st.info("아직 거래 내역이 없습니다")


# ══════════════════════════════════════════════════════════════
# Row 4: 가격 차트 (거래 내역 아래로 이동)
# ══════════════════════════════════════════════════════════════
st.subheader("🕯️ 가격 차트")

if candles_df is not None and not candles_df.empty:
    # timestamp 컬럼 또는 인덱스를 UTC → KST로 변환 (Pandas 2.0+ 호환)
    if 'timestamp' in candles_df.columns:
        x_data = pd.to_datetime(candles_df['timestamp'])
    elif isinstance(candles_df.index, pd.DatetimeIndex):
        x_data = candles_df.index
    else:
        x_data = pd.to_datetime(candles_df.index)

    # UTC → KST (+9시간)
    try:
        x_data_kst = x_data + pd.Timedelta(hours=9)
    except Exception:
        x_data_kst = x_data  # 변환 실패 시 원본 사용

    fig = go.Figure()

    # 이상치 필터링: 플래시 크래시 시 극단적 꼬리(wick) 클램핑
    # 가격 중심값 대비 ±3% 로 high/low 제한 (1분봉 기준 충분한 여유)
    _display_df = candles_df.copy()
    _body_high = _display_df[['open', 'close']].max(axis=1)
    _body_low = _display_df[['open', 'close']].min(axis=1)
    _mid = (_body_high + _body_low) / 2
    _max_wick = _mid * 0.03  # 가격의 3%
    _display_df['high'] = _display_df['high'].clip(upper=_body_high + _max_wick)
    _display_df['low'] = _display_df['low'].clip(lower=_body_low - _max_wick)

    # 캔들스틱
    fig.add_trace(go.Candlestick(
        x=x_data_kst,
        open=_display_df['open'],
        high=_display_df['high'],
        low=_display_df['low'],
        close=_display_df['close'],
        name="가격"
    ))

    # 선택된 거래의 진입/청산 시점 표시
    if selected_trade:
        entry_time_str = selected_trade.get('entry_time')
        exit_time_str = selected_trade.get('exit_time')
        entry_price = selected_trade.get('entry_price')
        exit_price = selected_trade.get('exit_price')
        trade_side = selected_trade.get('side')

        # 시간을 datetime으로 변환 후 KST로 변환 (Pandas 2.0+ 호환)
        try:
            def _to_kst(val):
                """다양한 형태의 시간값을 KST Timestamp로 변환"""
                if val is None:
                    return None
                if isinstance(val, (int, float)):
                    # Unix ms/s 정수 → Timestamp 변환
                    ts = pd.to_datetime(val, unit='ms') if val > 1e12 else pd.to_datetime(val, unit='s')
                else:
                    ts = pd.to_datetime(val)
                return ts + pd.Timedelta(hours=9)

            entry_dt = _to_kst(entry_time_str)
            exit_dt = _to_kst(exit_time_str)

            # 진입 시점 마커 (초록색 삼각형)
            entry_color = "lime" if trade_side == "LONG" else "red"
            exit_color = "red" if trade_side == "LONG" else "lime"

            # 진입 수직선
            if entry_dt is not None:
                fig.add_vline(
                    x=entry_dt,
                    line_dash="dash",
                    line_color=entry_color,
                    line_width=2,
                    annotation_text=f"진입 {entry_price:,.2f}",
                    annotation_position="top"
                )
                fig.add_hline(
                    y=entry_price,
                    line_dash="dot",
                    line_color=entry_color,
                    opacity=0.5
                )

            # 청산 수직선
            if exit_dt is not None and exit_price:
                fig.add_vline(
                    x=exit_dt,
                    line_dash="dash",
                    line_color=exit_color,
                    line_width=2,
                    annotation_text=f"청산 {exit_price:,.2f}",
                    annotation_position="top"
                )
                fig.add_hline(
                    y=exit_price,
                    line_dash="dot",
                    line_color=exit_color,
                    opacity=0.5
                )

        except Exception as e:
            st.warning(f"거래 시점 표시 중 오류: {e}")

    # 포지션 진입가 수평선
    if position and position.get('entry_price'):
        entry_p = position['entry_price']
        fig.add_hline(
            y=entry_p,
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"현재 진입가: {entry_p:,.2f}",
            annotation_position="top left"
        )

    # 손절/익절 라인
    if position:
        sl = position.get('stop_loss')
        tp = position.get('take_profit')
        if sl:
            fig.add_hline(y=sl, line_dash="dot", line_color="red",
                          annotation_text=f"SL: {sl:,.2f}")
        if tp:
            fig.add_hline(y=tp, line_dash="dot", line_color="green",
                          annotation_text=f"TP: {tp:,.2f}")

    # 최신 캔들 정보 표시
    latest_time = x_data_kst.iloc[-1] if len(x_data_kst) > 0 else None
    latest_close = candles_df['close'].iloc[-1] if len(candles_df) > 0 else 0

    chart_title = f"{symbol} 실시간 차트 (캔들: {len(candles_df)}개, 최신: {latest_time.strftime('%H:%M:%S') if latest_time else 'N/A'})"
    if selected_trade:
        chart_title += f" | 선택된 거래: {selected_trade.get('side')} ({selected_trade.get('pnl', 0):+.2f} USDT)"

    fig.update_layout(
        title=chart_title,
        xaxis_title="시간 (KST)",
        yaxis_title="가격 (USDT)",
        height=500,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # 고유 key로 캐싱 방지 (data_age 사용)
    chart_key = f"price_chart_{data_age:.0f}" if data_age else "price_chart_default"
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

    # 차트 아래에 캔들 범위 표시
    if len(x_data_kst) > 0:
        st.caption(f"📊 차트 범위: {x_data_kst.iloc[0].strftime('%Y-%m-%d %H:%M')} ~ {x_data_kst.iloc[-1].strftime('%Y-%m-%d %H:%M')} (KST)")
else:
    st.info("캔들 데이터를 불러오는 중...")


# ══════════════════════════════════════════════════════════════
# Row 4: 트레이딩 통계
# ══════════════════════════════════════════════════════════════
st.subheader("📊 트레이딩 통계")

stats = status.get('stats', {})
scol1, scol2, scol3, scol4, scol5, scol6 = st.columns(6)

total_trades = stats.get('total_trades', 0)
scol1.metric("총 거래", f"{total_trades}회")

winning = stats.get('winning_trades', 0)
scol2.metric("승리", f"{winning}회")

losing = stats.get('losing_trades', 0)
scol3.metric("패배", f"{losing}회")

win_rate = stats.get('win_rate', 0)
scol4.metric("승률", f"{win_rate:.1f}%")

max_dd = stats.get('max_drawdown', 0)
scol5.metric("최대 낙폭", f"{max_dd:.1%}")

consec_losses = stats.get('consecutive_losses', 0)
if consec_losses >= 3:
    scol6.metric("연속 손실", f"⚠️ {consec_losses}회")
else:
    scol6.metric("연속 손실", f"{consec_losses}회")

# ══════════════════════════════════════════════════════════════
# Row 5: 놓친 기회 (Missed Opportunities)
# ══════════════════════════════════════════════════════════════
st.subheader("🚨 놓친 기회")

if missed_opps:
    mo_df = pd.DataFrame(missed_opps)

    # 최신순 정렬
    mo_df = mo_df.iloc[::-1].reset_index(drop=True)

    # 표시용 컬럼 구성
    display_rows = []
    for _, row in mo_df.iterrows():
        direction = row.get('direction', '-')
        dir_icon = '📈' if direction == '상승' else '📉'
        swing = row.get('swing_pct', row.get('pct_change', 0))
        display_rows.append({
            '시간': row.get('time', '-')[:19],  # ISO → 초까지만
            '방향': f"{dir_icon} {direction}",
            '변동폭': f"{swing:+.2f}%",
            '고가': row.get('peak_high', '-'),
            '저가': row.get('peak_low', '-'),
            'Cascade': f"{row.get('cascade_conf', 0):.0%}",
            'Fee EV': f"{row.get('fee_ev_ratio', 0):.1f}x",
            '차단 사유': row.get('block_reason', '-'),
        })

    st.dataframe(
        pd.DataFrame(display_rows),
        use_container_width=True,
        height=min(300, 50 + len(display_rows) * 35),
    )

    st.caption(f"총 {len(missed_opps)}건 기록 (최근 50건 유지)")
else:
    st.info("놓친 기회 기록이 없습니다")


# ══════════════════════════════════════════════════════════════
# 사이드바: AI & 시스템 정보
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    cycle_ctx = status.get('cycle_ctx')
    _is_unified = cycle_ctx.get('use_unified_model', False) if cycle_ctx else False
    _unified_stats = status.get('model_stats', {}).get('unified')

    if _is_unified:
        # ══════════════════════════════════════════════════════════
        # 통합 모델 모드 사이드바
        # ══════════════════════════════════════════════════════════
        st.header("🧬 통합 모델 판단")

        if cycle_ctx:
            c_dir = cycle_ctx.get('cascade_dir', '-')
            final = cycle_ctx.get('final_score', 0)
            action = cycle_ctx.get('action', 'HOLD')
            exec_mode = cycle_ctx.get('execution_mode')

            dir_map = {'LONG': '🟢 LONG', 'SHORT': '🔴 SHORT', 'NEUTRAL': '⚪ NEUTRAL'}
            st.metric("예측 방향", dir_map.get(c_dir, f"⚪ {c_dir}"))
            st.metric("최종 스코어", f"{final:+.3f}")
            st.metric("액션", action)
            if exec_mode:
                mode_map = {'aggressive_taker': '⚡ 시장가', 'passive_maker': '🎯 지정가', 'fallback_market': '📦 기본'}
                st.metric("체결 모드", mode_map.get(exec_mode, exec_mode))

            # AI 성숙도
            ai_maturity = cycle_ctx.get('ai_maturity')
            ai_accuracy = cycle_ctx.get('ai_accuracy')
            pred_count = cycle_ctx.get('pred_count', 0)
            if ai_maturity is not None:
                if ai_maturity < 0.1:
                    mat_icon = "🔴 차단"
                elif ai_maturity < 0.5:
                    mat_icon = "🟡 감쇠"
                else:
                    mat_icon = "🟢 정상"
                st.metric("AI 성숙도", f"{ai_maturity:.0%}", delta=mat_icon)
                st.caption(f"정확도: {ai_accuracy:.1%} | 예측: {pred_count}건")

            # Regime / 변동성
            regime = cycle_ctx.get('regime', '-')
            regime_conf = cycle_ctx.get('regime_conf')
            vol_z = cycle_ctx.get('vol_z')
            vol_factor = cycle_ctx.get('vol_factor')
            st.caption(f"Regime: {regime}" + (f" ({regime_conf:.0%})" if regime_conf else ""))
            if vol_z is not None:
                st.caption(f"변동성: z={vol_z:+.2f} factor={vol_factor:.2f}" if vol_factor else f"변동성: z={vol_z:+.2f}")
        else:
            st.caption("사이클 데이터 대기 중")

        st.divider()

        # ── Shadow Mode 상태 ──
        st.header("🛡️ Shadow Mode")
        if _unified_stats:
            _sh_state = _unified_stats.get('shadow_state', '-')
            _sh_map = {'SHADOW': '⏳ 가상 거래 중', 'LIVE': '🟢 실전 활성', 'DEMOTED': '🔴 강등됨'}
            st.metric("Shadow 상태", _sh_map.get(_sh_state, _sh_state))
            _sh_resolved = _unified_stats.get('shadow_resolved', 0)
            _sh_acc = _unified_stats.get('shadow_accuracy', 0)
            _sh_pf = _unified_stats.get('shadow_profit_factor', 0)
            st.caption(f"해소: {_sh_resolved}건 | 정확도: {_sh_acc:.1%} | PF: {_sh_pf:.2f}")
            _sh_prom = _unified_stats.get('shadow_promotions', 0)
            _sh_dem = _unified_stats.get('shadow_demotions', 0)
            if _sh_prom or _sh_dem:
                st.caption(f"승격: {_sh_prom}회 | 강등: {_sh_dem}회")
        else:
            st.caption("Shadow 데이터 없음")

        st.divider()

        # ── MoE 전문가 라우팅 ──
        st.header("🎯 MoE 라우팅")
        if _unified_stats:
            _sharpness = _unified_stats.get('router_sharpness', 0)
            if _sharpness > 0.5:
                _sharp_icon = "🟢 선택적"
            elif _sharpness > 0.2:
                _sharp_icon = "🟡 보통"
            else:
                _sharp_icon = "🔴 균일"
            st.metric("Router Sharpness", f"{_sharpness:.2f}", delta=_sharp_icon)

            _ew = _unified_stats.get('expert_weights', [])
            if _ew:
                _expert_names = ['Macro', 'Mid', 'Micro', 'Cross']
                _ew_parts = []
                for _ei, _w in enumerate(_ew):
                    _en = _expert_names[_ei] if _ei < len(_expert_names) else f"E{_ei}"
                    _ew_parts.append(f"{_en}={_w:.0%}")
                st.caption(" | ".join(_ew_parts))

            _moe_comb = _unified_stats.get('moe_combined')
            _moe_mult = _unified_stats.get('moe_size_mult')
            if _moe_comb is not None:
                st.metric("MoE 확신도", f"{_moe_comb:.2f}", delta=f"사이즈 {_moe_mult:.2f}x" if _moe_mult else None)
        else:
            st.caption("MoE 데이터 없음")

    else:
        # ══════════════════════════════════════════════════════════
        # 레거시 MTF Cascade 모드 사이드바
        # ══════════════════════════════════════════════════════════
        st.header("🔥 Cascade 판단")

        if cycle_ctx:
            c_dir = cycle_ctx.get('cascade_dir', '-')
            c_conf = cycle_ctx.get('cascade_conf', 0)
            c_fire = cycle_ctx.get('cascade_fire', False)
            fee_ev = cycle_ctx.get('fee_ev_ratio', 0)
            final = cycle_ctx.get('final_score', 0)
            action = cycle_ctx.get('action', 'HOLD')
            exec_mode = cycle_ctx.get('execution_mode')

            dir_map = {'LONG': '🟢 LONG', 'SHORT': '🔴 SHORT', 'NEUTRAL': '⚪ NEUTRAL'}
            fire_icon = "🔥 발화" if c_fire else "⏳ 미발화"

            st.metric("Cascade 방향", dir_map.get(c_dir, f"⚪ {c_dir}"))
            st.metric("Cascade 신뢰도", f"{c_conf:.1%}", delta=fire_icon)
            st.metric("Fee EV", f"{fee_ev:.1f}x", delta="통과" if fee_ev >= 1.2 else "미달")
            st.metric("최종 스코어", f"{final:+.3f}")
            st.metric("액션", action)
            if exec_mode:
                mode_map = {'aggressive_taker': '⚡ 시장가', 'passive_maker': '🎯 지정가', 'fallback_market': '📦 기본'}
                st.metric("체결 모드", mode_map.get(exec_mode, exec_mode))

            # AI 성숙도
            ai_maturity = cycle_ctx.get('ai_maturity')
            ai_accuracy = cycle_ctx.get('ai_accuracy')
            pred_count = cycle_ctx.get('pred_count', 0)
            if ai_maturity is not None:
                if ai_maturity < 0.1:
                    mat_icon = "🔴 차단"
                elif ai_maturity < 0.5:
                    mat_icon = "🟡 감쇠"
                else:
                    mat_icon = "🟢 정상"
                st.metric("AI 성숙도", f"{ai_maturity:.0%}", delta=mat_icon)
                st.caption(f"정확도: {ai_accuracy:.1%} | 예측: {pred_count}건")

            # Regime / 변동성
            regime = cycle_ctx.get('regime', '-')
            regime_conf = cycle_ctx.get('regime_conf')
            vol_z = cycle_ctx.get('vol_z')
            vol_factor = cycle_ctx.get('vol_factor')
            st.caption(f"Regime: {regime}" + (f" ({regime_conf:.0%})" if regime_conf else ""))
            if vol_z is not None:
                st.caption(f"변동성: z={vol_z:+.2f} factor={vol_factor:.2f}" if vol_factor else f"변동성: z={vol_z:+.2f}")
        else:
            st.caption("Cascade 데이터 없음 (첫 사이클 대기)")

        st.divider()

        # ── 2. 모델별 상세 (레거시 MTF) ──
        st.header("🧠 모델별 판단")

        ai_info = status.get('ai_info')
        if ai_info:
            mtf_preds = ai_info.get('mtf_predictions', {})

            if mtf_preds:
                st.markdown("**🌐 Cascade 입력 (MTF)**")
                role_map = {
                    '15m': 'Step1 추세',
                    '5m': 'Step2 모멘텀',
                    '3m': 'Step2 모멘텀',
                }
                for itv in ['15m', '5m', '3m']:
                    pred = mtf_preds.get(itv)
                    if pred:
                        d = pred.get('direction', '-')
                        c = pred.get('confidence', 0)
                        icon = '🟢' if d == 'UP' else ('🔴' if d == 'DOWN' else '⚪')
                        role = role_map.get(itv, '')
                        st.caption(f"  {icon} {itv} ({role}): {d} {c:.0%}")
            else:
                st.caption("MTF 데이터 없음")

            st.markdown("---")

            # 1분봉 딥러닝 모델 (Cascade Step3)
            st.markdown("**🔬 1m Transformer (Step3 미시필터)**")
            direction = ai_info.get('direction', '-')
            dir_emoji = '🟢' if direction == 'UP' else ('🔴' if direction == 'DOWN' else '⚪')
            conf = ai_info.get('confidence', 0)
            change_pct = ai_info.get('price_change_pct')
            st.caption(f"  {dir_emoji} {direction} ({conf:.0%})")
            if change_pct is not None:
                st.caption(f"  예상 변화: {change_pct:+.2f}%")

            st.markdown("---")

            # 10초 틱 모델 (Cascade Step4)
            st.markdown("**⚡ 10s Tick (Step4 실행게이트)**")
            tick_dir = ai_info.get('tick_direction')
            tick_timing = ai_info.get('tick_timing')
            tick_dir_conf = ai_info.get('tick_direction_conf')
            if tick_dir:
                t_icon = '🟢' if tick_dir == 'UP' else ('🔴' if tick_dir == 'DOWN' else '⚪')
                st.caption(f"  {t_icon} {tick_dir}" + (f" ({tick_dir_conf:.0%})" if tick_dir_conf else ""))
                if tick_timing is not None:
                    st.caption(f"  타이밍 신뢰도: {tick_timing:.0%}")
            else:
                st.caption("  틱 데이터 없음")
        else:
            st.caption("AI 모델 데이터 없음")

    st.divider()

    # ── 3. 데이터 상태 ─────────────────────────────────────────
    st.header("📡 데이터 상태")
    data_status = status.get('data_status')
    if data_status:
        st.metric("캔들 수", data_status.get('candle_count', 0))
        st.metric("현재가", f"{data_status.get('current_price', 0):,.2f}")
        st.metric("마크가", f"{data_status.get('mark_price', 0):,.2f}")
        st.metric("펀딩비율", f"{data_status.get('funding_rate', 0):.4%}")

        connected = data_status.get('is_connected', False)
        st.metric("WebSocket", "🟢 연결됨" if connected else "🔴 끊김")
        st.metric("수신 메시지", f"{data_status.get('message_count', 0):,}")
    else:
        st.caption("데이터 상태 없음")

    st.divider()

    # ── 4. 시스템 ──────────────────────────────────────────────
    st.header("⚙️ 시스템")
    st.caption(f"봇 상태: {bot_state}")
    st.caption(f"심볼: {symbol}")
    st.caption(f"레버리지: {default_lev}x → {current_lev}x")
    if data_age is not None:
        st.caption(f"데이터 경과: {data_age:.1f}초")
