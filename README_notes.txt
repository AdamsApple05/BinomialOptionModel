CRR Surface Delta-Hedge Research Framework

Purpose:
- Compare a CRR-based cross-sectional relative-value strategy in:
  1) a bucket where CRR historically fit best
  2) a bucket where CRR historically fit poorly

Buckets:
- PUT_30_60_OTM
- CALL_120_150_ATM

Signal:
- Build same-day implied vol surface
- Compute surface-fair CRR price
- residual = fair - market
- positive residual -> buy cheap option
- negative residual -> short rich option

Hedge:
- Delta hedge with underlying SPY shares using CRR delta.

Important:
- This uses daily data only.
- Bid/ask are synthesized from close.
- It is a research backtester, not a live trading engine.
