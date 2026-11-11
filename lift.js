// The lift arithmetic, one copy for the page and for the check run under node.
// Same estimator as the paper: variance explained over rows, the last axis is the feature axis.
(function (root) {
  "use strict";

  // Parse a NumPy .npy file (format 1.0-3.0, little-endian float16/32/64, C order) to {shape, data}.
  function parseNpy(buf) {
    const u8 = new Uint8Array(buf);
    if (u8[0] !== 0x93 || String.fromCharCode(...u8.slice(1, 6)) !== "NUMPY") throw new Error("not a .npy file");
    const major = u8[6], dv = new DataView(buf);
    const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
    const start = major === 1 ? 10 : 12;
    const header = new TextDecoder("latin1").decode(u8.slice(start, start + hlen));
    const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
    if (/'fortran_order':\s*True/.test(header)) throw new Error("Fortran-ordered arrays are not supported; save with np.ascontiguousarray");
    const shape = /'shape':\s*\(([^)]*)\)/.exec(header)[1].split(",").map(s => s.trim()).filter(Boolean).map(Number);
    const off = start + hlen, n = shape.reduce((a, b) => a * b, 1), body = buf.slice(off);
    let data;
    if (descr === "<f8") data = new Float64Array(body, 0, n);
    else if (descr === "<f4") data = new Float32Array(body, 0, n);
    else if (descr === "<f2") data = halfToFloat(new Uint16Array(body, 0, n));
    else throw new Error(`dtype ${descr} is not supported; save as float32`);
    return { shape, data };
  }

  function halfToFloat(h) {
    const out = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) {
      const s = h[i] & 0x8000 ? -1 : 1, e = (h[i] >> 10) & 0x1f, f = h[i] & 0x3ff;
      out[i] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
    }
    return out;
  }

  // 1 - mean_rows ||p - t||^2 / mean_rows ||t - mean(t)||^2 over rows [lo, hi) of width d.
  function varianceExplained(p, t, d, lo, hi) {
    const mean = new Float64Array(d), rows = hi - lo;
    for (let r = lo; r < hi; r++) for (let j = 0; j < d; j++) mean[j] += t[r * d + j];
    for (let j = 0; j < d; j++) mean[j] /= rows;
    let resid = 0, spread = 0;
    for (let r = lo; r < hi; r++) for (let j = 0; j < d; j++) {
      const i = r * d + j, e = p[i] - t[i], c = t[i] - mean[j];
      resid += e * e; spread += c * c;
    }
    return 1 - (resid / rows) / Math.max(spread / rows, 1e-12);
  }

  // 0 at the floor, 1 at the ceiling; NaN when the span is too narrow to divide by.
  function lifted(value, floor, ceiling = 1) {
    if (![value, floor, ceiling].every(Number.isFinite)) return NaN;
    if (ceiling - floor < Math.max(1e-6, 1e-4 * Math.max(Math.abs(floor), Math.abs(ceiling)))) return NaN;
    return (value - floor) / (ceiling - floor);
  }

  // pred, target, last: {shape, data} of one shape; returns value, floor, lift, and each half's lift.
  function check(pred, target, last) {
    const same = a => a.shape.length === target.shape.length && a.shape.every((s, i) => s === target.shape[i]);
    if (!same(pred) || !same(last)) throw new Error(`shapes differ: ${pred.shape} / ${target.shape} / ${last.shape}`);
    const d = target.shape.length > 1 ? target.shape[target.shape.length - 1] : 1;
    const rows = target.data.length / d, half = Math.floor(rows / 2);
    const one = (lo, hi) => {
      const value = varianceExplained(pred.data, target.data, d, lo, hi);
      const floor = varianceExplained(last.data, target.data, d, lo, hi);
      return { value, floor, lift: lifted(value, floor) };
    };
    return { ...one(0, rows), rows, d, halves: rows >= 4 ? [one(0, half).lift, one(half, rows).lift] : null };
  }

  const api = { parseNpy, varianceExplained, lifted, check };
  if (typeof module !== "undefined") module.exports = api; else root.Lift = api;
})(this);
