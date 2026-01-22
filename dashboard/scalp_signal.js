// Scalp signal/backtest module (detached from UI until re-enabled).
// This file keeps the last prototype logic for reference and future wiring.
// Exported as a plain object to avoid bundler requirements.
const ScalpSignal = (() => {
  const FEE_BPS = 2;

  function buildScalpDrawings(klines, reverse) {
    const output = [];
    const byBucket = new Map();
    for (const k of klines || []) {
      const t = Number(k.time || 0);
      if (!Number.isFinite(t) || t <= 0) continue;
      const hi = Number(k.high);
      const lo = Number(k.low);
      const cl = Number(k.close);
      if (!Number.isFinite(hi) || !Number.isFinite(lo) || !Number.isFinite(cl)) continue;
      const bucket = Math.floor(t / 300) * 300;
      if (!byBucket.has(bucket)) byBucket.set(bucket, []);
      byBucket.get(bucket).push({ ...k, time: t, high: hi, low: lo, close: cl });
    }
    const buckets = Array.from(byBucket.keys()).sort((a, b) => a - b);
    let activeTrade = null;
    for (const bucket of buckets) {
      const group = (byBucket.get(bucket) || []).slice().sort((a, b) => a.time - b.time);
      if (group.length < 5) continue;
      if (activeTrade) {
        for (const candle of group) {
          if (!candle) continue;
          if (activeTrade.dir === "long") {
            if (candle.low <= activeTrade.sl || candle.high >= activeTrade.tp) {
              activeTrade = null;
              break;
            }
          } else {
            if (candle.high >= activeTrade.sl || candle.low <= activeTrade.tp) {
              activeTrade = null;
              break;
            }
          }
        }
        if (activeTrade) continue;
      }
      let top = -Infinity;
      let bottom = Infinity;
      for (let i = 0; i < 5; i += 1) {
        const c = group[i];
        if (!c) continue;
        if (c.high > top) top = c.high;
        if (c.low < bottom) bottom = c.low;
      }
      if (!Number.isFinite(top) || !Number.isFinite(bottom)) continue;
      const nextBucket = bucket + 5 * 60;
      const nextGroup = (byBucket.get(nextBucket) || []).slice().sort((a, b) => a.time - b.time);
      if (nextGroup.length < 2) continue;
      const window = nextGroup.slice(0, 4);
      let signal = null;
      for (let i = 0; i < window.length; i += 1) {
        const candle = window[i];
        if (!candle) continue;
        if (candle.high > top) {
          signal = { dir: "long", breakout: candle, entry: nextGroup[i + 1] };
          break;
        }
        if (candle.low < bottom) {
          signal = { dir: "short", breakout: candle, entry: nextGroup[i + 1] };
          break;
        }
      }
      if (!signal || !signal.entry) continue;
      const entry = Number(signal.entry.open);
      const entryTime = signal.entry.time;
      if (!Number.isFinite(entry) || !Number.isFinite(entryTime)) continue;

      const baseDir = signal.dir;
      let baseSL = 0;
      let baseTP = 0;
      if (baseDir === "long") {
        baseSL = signal.breakout.low * (1 - 0.0002);
        const risk = entry - baseSL;
        baseTP = entry + risk * 1.5;
      } else {
        baseSL = signal.breakout.high * (1 + 0.0002);
        const risk = baseSL - entry;
        baseTP = entry - risk * 1.5;
      }
      const finalDir = reverse ? (baseDir === "long" ? "short" : "long") : baseDir;
      const sl = reverse ? baseTP : baseSL;
      const tp = reverse ? baseSL : baseTP;
      activeTrade = { dir: finalDir, sl, tp };
      output.push({ type: "signal", dir: finalDir, entry, entryTime, sl, tp });
    }
    return output;
  }

  function runScalpBacktest(klines, reverse) {
    const trades = [];
    if (!Array.isArray(klines) || klines.length < 10) return trades;
    const byBucket = new Map();
    for (const k of klines) {
      const t = Number(k.time || 0);
      if (!Number.isFinite(t) || t <= 0) continue;
      const o = Number(k.open);
      const h = Number(k.high);
      const l = Number(k.low);
      const c = Number(k.close);
      if (![o, h, l, c].every(Number.isFinite)) continue;
      const bucket = Math.floor(t / 300) * 300;
      if (!byBucket.has(bucket)) byBucket.set(bucket, []);
      byBucket.get(bucket).push({ time: t, open: o, high: h, low: l, close: c });
    }
    const buckets = Array.from(byBucket.keys()).sort((a, b) => a - b);
    let active = null;
    for (const bucket of buckets) {
      const group = (byBucket.get(bucket) || []).slice().sort((a, b) => a.time - b.time);
      if (group.length < 5) continue;
      if (active) {
        for (const candle of group) {
          if (active.dir === "long") {
            if (candle.low <= active.sl) {
              trades.push({ ...active, exit: active.sl, exitTime: candle.time, result: "SL" });
              active = null;
              break;
            }
            if (candle.high >= active.tp) {
              trades.push({ ...active, exit: active.tp, exitTime: candle.time, result: "TP" });
              active = null;
              break;
            }
          } else {
            if (candle.high >= active.sl) {
              trades.push({ ...active, exit: active.sl, exitTime: candle.time, result: "SL" });
              active = null;
              break;
            }
            if (candle.low <= active.tp) {
              trades.push({ ...active, exit: active.tp, exitTime: candle.time, result: "TP" });
              active = null;
              break;
            }
          }
        }
        if (active) continue;
      }
      let top = -Infinity;
      let bottom = Infinity;
      for (let i = 0; i < 5; i += 1) {
        const c = group[i];
        if (!c) continue;
        if (c.high > top) top = c.high;
        if (c.low < bottom) bottom = c.low;
      }
      if (!Number.isFinite(top) || !Number.isFinite(bottom)) continue;
      const nextBucket = bucket + 5 * 60;
      const nextGroup = (byBucket.get(nextBucket) || []).slice().sort((a, b) => a.time - b.time);
      if (nextGroup.length < 2) continue;
      const window = nextGroup.slice(0, 4);
      let signal = null;
      for (let i = 0; i < window.length; i += 1) {
        const candle = window[i];
        if (!candle) continue;
        if (candle.high > top) {
          signal = { dir: "long", breakout: candle, entry: nextGroup[i + 1] };
          break;
        }
        if (candle.low < bottom) {
          signal = { dir: "short", breakout: candle, entry: nextGroup[i + 1] };
          break;
        }
      }
      if (!signal || !signal.entry) continue;
      const entry = signal.entry.open;
      const entryTime = signal.entry.time;
      let baseSL = 0;
      let baseTP = 0;
      if (signal.dir === "long") {
        baseSL = signal.breakout.low * (1 - 0.0002);
        const risk = entry - baseSL;
        baseTP = entry + risk * 1.5;
      } else {
        baseSL = signal.breakout.high * (1 + 0.0002);
        const risk = baseSL - entry;
        baseTP = entry - risk * 1.5;
      }
      const finalDir = reverse ? (signal.dir === "long" ? "short" : "long") : signal.dir;
      const sl = reverse ? baseTP : baseSL;
      const tp = reverse ? baseSL : baseTP;
      active = {
        dir: finalDir,
        entry,
        entryTime,
        sl,
        tp
      };
    }
    return trades;
  }

  return { buildScalpDrawings, runScalpBacktest, FEE_BPS };
})();

window.ScalpSignal = ScalpSignal;
