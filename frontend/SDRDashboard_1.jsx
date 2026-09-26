import React, { useRef, useEffect, useState, useCallback, useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceArea, ReferenceDot, Area, AreaChart
} from "recharts";
import { Upload, FileAudio, ChevronRight, Radio, AlertTriangle } from "lucide-react";

/* ============================================================================
   SDR SIGNAL ANALYSIS DASHBOARD
   -------------------------------------------------------------------------
   Expects the backend result shaped as:
   {
     fs, rs,
     modulation: { class, confidence, cumulants: { c20, c21, c40, c41, c42, ... } },
     interleaver: { type, width, confidence, rank_profile: [{width, rank}] },
     fec: { type, params, confidence },
     frames: { length, preamble_matches, hex_dump: [{offset, hex, ascii, region}] }
   }

   Raw IQ / STFT / constellation arrays are NOT part of that JSON — the
   backend pipeline should additionally expose lightweight derived arrays
   (a decimated STFT magnitude grid, a PSD array, and pre/post-sync IQ
   scatter samples) alongside the result object. Integration points are
   marked INTEGRATION below. Until then, this file runs on a self-contained
   mock generator so it renders and demos standalone.
   ========================================================================== */

// ---------------------------------------------------------------------------
// Design tokens (Crisp Light Mode Palette)
// ---------------------------------------------------------------------------
const T = {
  void: "transparent",
  bgGradient: "linear-gradient(135deg, #00F260 0%, #0575E6 100%)",
  panel: "#FFFFFF",
  panelAlt: "#F8FAFC",
  panelHeader: "#F0F9FF",
  border: "#E2E8F0",
  borderBright: "#CBD5E1",
  hover: "#F0F7FF",
  phosphor: "#059669",
  phosphorDim: "#10B981",
  amber: "#D97706",
  amberDim: "#F59E0B",
  cyan: "#0284C7",
  red: "#DC2626",
  violet: "#7C3AED",
  text: "#334155",
  textDim: "#64748B",
  textFaint: "#94A3B8",
};

const mono = { fontFamily: "'Plus Jakarta Sans', system-ui, -apple-system, sans-serif" };
const dataMono = { fontFamily: "'JetBrains Mono','IBM Plex Mono',ui-monospace,'Courier New',monospace" };

function useGoogleFont() {
  useEffect(() => {
    const id = "sdr-dashboard-font-jakarta";
    if (document.getElementById(id)) return;
    const preconnect1 = document.createElement("link");
    preconnect1.rel = "preconnect";
    preconnect1.href = "https://fonts.googleapis.com";
    const preconnect2 = document.createElement("link");
    preconnect2.rel = "preconnect";
    preconnect2.href = "https://fonts.gstatic.com";
    preconnect2.crossOrigin = "anonymous";
    const link = document.createElement("link");
    link.id = id;
    link.rel = "stylesheet";
    link.href =
      "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap";
    document.head.append(preconnect1, preconnect2, link);
  }, []);
}

// ---------------------------------------------------------------------------
// Mock data generation (INTEGRATION: replace with real pipeline output)
// ---------------------------------------------------------------------------
function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

function buildMockResult() {
  const rnd = seededRandom(42);
  const hexRegions = [
    { region: "preamble", len: 4 },
    { region: "header", len: 3 },
    { region: "payload", len: 10 },
    { region: "checksum", len: 2 },
  ];
  const hex_dump = [];
  let offset = 0;
  hexRegions.forEach(({ region, len }) => {
    for (let i = 0; i < len; i++) {
      const bytes = Array.from({ length: 8 }, () =>
        Math.floor(rnd() * 256).toString(16).padStart(2, "0")
      );
      const ascii = bytes
        .map((b) => {
          const c = parseInt(b, 16);
          return c >= 32 && c <= 126 ? String.fromCharCode(c) : ".";
        })
        .join("");
      hex_dump.push({ offset, hex: bytes.join(" "), ascii, region });
      offset += 8;
    }
  });

  const rank_profile = Array.from({ length: 15 }, (_, i) => {
    const width = i + 2;
    const isDip = width === 9;
    const base = 7.2 + rnd() * 0.6;
    return { width, rank: isDip ? 2.1 + rnd() * 0.3 : base };
  });

  return {
    fs: 2_400_000,
    rs: 250_000,
    modulation: {
      class: "8PSK",
      confidence: 0.93,
      cumulants: { c20: 0.02, c21: 1.01, c40: -0.98, c41: -0.02, c42: 0.34 },
    },
    interleaver: { type: "block", width: 9, confidence: 0.88, rank_profile },
    fec: { type: "convolutional", params: "K=7, R=1/2", confidence: 0.81 },
    frames: { length: 1176, preamble_matches: 14, hex_dump },
  };
}

function buildMockWaterfall(rows = 96, cols = 160) {
  const rnd = seededRandom(7);
  const data = [];
  for (let r = 0; r < rows; r++) {
    const row = new Float32Array(cols);
    const centerDrift = cols / 2 + Math.sin(r / 14) * 18;
    const bw = 14 + Math.sin(r / 30) * 3;
    for (let c = 0; c < cols; c++) {
      const noise = rnd() * 0.16;
      const d = Math.abs(c - centerDrift);
      const signal = d < bw ? Math.exp(-(d * d) / (2 * (bw / 2.3) ** 2)) : 0;
      row[c] = Math.min(1, noise + signal * (0.7 + rnd() * 0.3));
    }
    data.push(row);
  }
  return data;
}

function buildMockPSD(waterfall) {
  const cols = waterfall[0].length;
  const acc = new Float32Array(cols);
  waterfall.forEach((row) => row.forEach((v, c) => (acc[c] += v)));
  const n = waterfall.length;
  return Array.from({ length: cols }, (_, i) => {
    const freqMHz = ((i - cols / 2) / cols) * 2.4;
    const mag = acc[i] / n;
    const dB = 10 * Math.log10(Math.max(mag, 1e-4)) + 30;
    return { freq: Number(freqMHz.toFixed(3)), power: Number(dB.toFixed(2)) };
  });
}

function buildMockConstellation(cls, synced) {
  const rnd = seededRandom(synced ? 11 : 13);
  const orderMap = { BPSK: 2, QPSK: 4, "8PSK": 8, "16QAM": 16, "16APSK": 16 };
  const order = orderMap[cls] || 8;
  const pts = [];
  const n = 260;
  const phaseOffset = synced ? 0 : 0.9;
  const freqOffset = synced ? 0 : 0.014;
  const jitter = synced ? 0.05 : 0.22;
  for (let i = 0; i < n; i++) {
    const sym = i % order;
    const baseAngle = (2 * Math.PI * sym) / order;
    const drift = freqOffset * i + phaseOffset;
    const angle = baseAngle + drift + (rnd() - 0.5) * jitter;
    const r = 1 + (rnd() - 0.5) * (synced ? 0.08 : 0.3);
    pts.push({ x: r * Math.cos(angle), y: r * Math.sin(angle) });
  }
  return pts;
}

// ---------------------------------------------------------------------------
// Small structural bits
// ---------------------------------------------------------------------------
function CornerTicks() {
  const s = { position: "absolute", width: 8, height: 8, borderColor: T.borderBright };
  return (
    <>
      <div style={{ ...s, top: -1, left: -1, borderTop: "1px solid", borderLeft: "1px solid" }} />
      <div style={{ ...s, top: -1, right: -1, borderTop: "1px solid", borderRight: "1px solid" }} />
      <div style={{ ...s, bottom: -1, left: -1, borderBottom: "1px solid", borderLeft: "1px solid" }} />
      <div style={{ ...s, bottom: -1, right: -1, borderBottom: "1px solid", borderRight: "1px solid" }} />
    </>
  );
}

function Panel({ title, sub, accent = T.phosphor, right, children }) {
  return (
    <div
      style={{
        position: "relative",
        background: T.panel,
        border: `1px solid ${T.border}`,
        boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
        width: "100%",
        boxSizing: "border-box",
      }}
    >
      <CornerTicks />
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "8px 12px",
          background: T.panelHeader,
          borderBottom: `1px solid ${T.border}`,
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: "8px", minWidth: 0 }}>
          <span
            style={{
              width: "6px",
              height: "6px",
              flexShrink: 0,
              background: accent,
              boxShadow: `0 0 6px ${accent}`,
            }}
          />
          <span style={{ fontSize: "11px", letterSpacing: "0.025em", color: T.text, fontWeight: 600 }}>
            {title}
          </span>
          {sub && (
            <span style={{ fontSize: "10px", color: T.textDim }}>
              {sub}
            </span>
          )}
        </div>
        {right && <div style={{ flexShrink: 0 }}>{right}</div>}
      </div>
      <div style={{ padding: "12px" }}>{children}</div>
    </div>
  );
}

function ConfidenceBar({ value, accent = T.phosphor }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
      <div style={{ height: "6px", flex: 1, background: T.border, borderRadius: "2px", overflow: "hidden" }}>
        <div
          style={{ width: `${Math.round(value * 100)}%`, height: "100%", background: accent }}
        />
      </div>
      <span style={{ fontSize: "10px", width: "36px", textAlign: "right", color: T.text, fontWeight: 600 }}>
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 1. Upload panel
// ---------------------------------------------------------------------------
function UploadPanel({ fileName, onFile }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  };

  return (
    <Panel title="SIGNAL INPUT" accent={T.phosphor}>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "8px",
          cursor: "pointer",
          border: `1px dashed ${dragOver ? T.phosphor : T.borderBright}`,
          background: dragOver ? T.hover : "#FAFDFB",
          padding: "24px 12px",
          boxSizing: "border-box",
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".iq,.wav"
          style={{ display: "none" }}
          onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
        />
        {fileName ? (
          <>
            <FileAudio size={18} color={T.phosphor} />
            <span style={{ fontSize: "11px", fontWeight: 600, color: T.text }}>
              {fileName}
            </span>
            <span style={{ fontSize: "9px", color: T.textDim }}>
              DROP TO REPLACE
            </span>
          </>
        ) : (
          <>
            <Upload size={18} color={T.textDim} />
            <span style={{ fontSize: "11px", textAlign: "center", color: T.textDim, lineHeight: 1.4 }}>
              DROP .IQ / .WAV FILE
              <br />
              OR CLICK TO BROWSE
            </span>
          </>
        )}
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// 2. Classifier Reasoning
// ---------------------------------------------------------------------------
function explainModulation(mod) {
  const { class: cls, cumulants } = mod;
  const lines = [];
  if (cumulants.c42 !== undefined) {
    lines.push(
      `Normalized fourth-order cumulant C42 = ${cumulants.c42.toFixed(2)} falls in the band ` +
        `associated with ${cls} constellations rather than lower- or higher-order PSK/QAM neighbors.`
    );
  }
  if (cumulants.c40 !== undefined) {
    lines.push(
      `C40 = ${cumulants.c40.toFixed(2)} is close to the theoretical value for constant-envelope ` +
        `phase modulation, ruling out amplitude-varying schemes like QAM.`
    );
  }
  if (cumulants.c21 !== undefined) {
    lines.push(`C21 = ${cumulants.c21.toFixed(2)} confirms non-trivial signal power above the noise floor estimate.`);
  }
  return lines;
}

function ExplainabilityCard({ modulation, fec }) {
  const lines = explainModulation(modulation);
  return (
    <Panel title="CLASSIFIER REASONING" sub="modulation" accent={T.cyan}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
        <div>
          <div style={{ fontSize: "20px", fontWeight: 700, lineHeight: 1, color: T.cyan }}>
            {modulation.class}
          </div>
          <div style={{ fontSize: "9px", marginTop: "4px", color: T.textDim }}>
            DECISION CLASS
          </div>
        </div>
        <div style={{ width: "130px" }}>
          <ConfidenceBar value={modulation.confidence} accent={T.cyan} />
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "8px", marginBottom: "12px" }}>
        {Object.entries(modulation.cumulants).map(([k, v]) => (
          <div key={k} style={{ border: `1px solid ${T.border}`, padding: "6px 8px", background: "#F8FAFC" }}>
            <div style={{ fontSize: "8px", color: T.textDim }}>
              {k.toUpperCase()}
            </div>
            <div style={{ fontSize: "12px", fontWeight: 600, color: T.text }}>
              {v.toFixed(3)}
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "6px", marginBottom: "12px" }}>
        {lines.map((l, i) => (
          <div key={i} style={{ display: "flex", gap: "6px", fontSize: "10.5px", lineHeight: 1.4, color: T.textDim }}>
            <ChevronRight size={12} color={T.cyan} style={{ flexShrink: 0, marginTop: "2px" }} />
            <span>{l}</span>
          </div>
        ))}
      </div>

      <div style={{ paddingTop: "8px", borderTop: `1px solid ${T.border}` }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "4px" }}>
          <span style={{ fontSize: "9px", letterSpacing: "0.025em", color: T.textDim }}>
            FEC · {fec.type.toUpperCase()} ({fec.params})
          </span>
          <span style={{ fontSize: "10px", fontWeight: 600, color: T.amber }}>
            {(fec.confidence * 100).toFixed(0)}%
          </span>
        </div>
        <ConfidenceBar value={fec.confidence} accent={T.amber} />
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// 3. Waterfall (canvas)
// ---------------------------------------------------------------------------
function magToColor(v) {
  const t = Math.max(0, Math.min(1, v));
  if (t < 0.55) {
    const k = t / 0.55;
    return [Math.round(6 + k * 10), Math.round(10 + k * 60), Math.round(6 + k * 30)];
  }
  const k = (t - 0.55) / 0.45;
  return [
    Math.round(16 + k * 180),
    Math.round(70 + k * 185),
    Math.round(36 + k * 130),
  ];
}

function WaterfallPanel({ waterfall, fs }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !waterfall.length) return;
    const rows = waterfall.length;
    const cols = waterfall[0].length;
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(cols, rows);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const [rr, gg, bb] = magToColor(waterfall[r][c]);
        const idx = (r * cols + c) * 4;
        img.data[idx] = rr;
        img.data[idx + 1] = gg;
        img.data[idx + 2] = bb;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [waterfall]);

  return (
    <Panel
      title="WATERFALL"
      sub={`FS ${(fs / 1e6).toFixed(3)} MSPS`}
      accent={T.phosphor}
      right={<span style={{ fontSize: "9px", color: T.textDim }}>|FFT|&sup2; vs t</span>}
    >
      <div style={{ position: "relative", border: `1px solid ${T.border}` }}>
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height: 160, imageRendering: "pixelated", display: "block" }}
        />
        <div
          style={{
            position: "absolute",
            left: 4,
            right: 4,
            bottom: 2,
            display: "flex",
            justifyContent: "space-between",
            fontSize: "8px",
            color: "#A7F3D0",
            textShadow: "0 0 3px #000",
          }}
        >
          <span>-FS/2</span>
          <span>0</span>
          <span>+FS/2</span>
        </div>
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// 4. Welch PSD
// ---------------------------------------------------------------------------
function PSDPanel({ psd, rs }) {
  return (
    <Panel title="WELCH PSD" sub={`RS ${(rs / 1e3).toFixed(1)} KSPS`} accent={T.amber}>
      <div style={{ height: 200, width: "100%" }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={psd} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
            <defs>
              <linearGradient id="psdFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={T.amber} stopOpacity={0.25} />
                <stop offset="100%" stopColor={T.amber} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke={T.border} strokeDasharray="0" vertical={false} />
            <XAxis
              dataKey="freq"
              tick={{ fill: T.textDim, fontSize: 9 }}
              stroke={T.border}
              tickFormatter={(v) => `${v}`}
              label={{ value: "MHz", position: "insideBottomRight", fill: T.textDim, fontSize: 9, offset: -2 }}
            />
            <YAxis
              tick={{ fill: T.textDim, fontSize: 9 }}
              stroke={T.border}
              width={38}
              label={{ value: "dB", angle: -90, position: "insideLeft", fill: T.textDim, fontSize: 9 }}
            />
            <Tooltip
              contentStyle={{ background: T.panel, border: `1px solid ${T.border}`, fontSize: 11 }}
              labelStyle={{ color: T.textDim }}
              itemStyle={{ color: T.amber }}
            />
            <Area type="monotone" dataKey="power" stroke={T.amber} strokeWidth={1.25} fill="url(#psdFill)" isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// 5. Constellation
// ---------------------------------------------------------------------------
function ConstellationCanvas({ points, accent }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const size = 200;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size, size);

    ctx.strokeStyle = T.border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(size / 2, 0); ctx.lineTo(size / 2, size);
    ctx.moveTo(0, size / 2); ctx.lineTo(size, size / 2);
    ctx.stroke();
    ctx.strokeStyle = T.borderBright;
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, size / 2 - 10, 0, Math.PI * 2);
    ctx.stroke();

    const scale = (size / 2 - 16) / 1.4;
    points.forEach((p) => {
      const x = size / 2 + p.x * scale;
      const y = size / 2 - p.y * scale;
      ctx.fillStyle = accent;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.globalAlpha = 1;
  }, [points, accent]);
  return <canvas ref={ref} style={{ width: 200, height: 200, display: "block", margin: "0 auto" }} />;
}

function ConstellationPanel({ cls }) {
  const [mode, setMode] = useState("split");
  const raw = useMemo(() => buildMockConstellation(cls, false), [cls]);
  const synced = useMemo(() => buildMockConstellation(cls, true), [cls]);

  const Toggle = ({ id, label }) => (
    <button
      onClick={() => setMode(id)}
      style={{
        padding: "2px 8px",
        fontSize: "9px",
        letterSpacing: "0.025em",
        border: `1px solid ${mode === id ? T.phosphor : T.border}`,
        color: mode === id ? T.phosphor : T.textDim,
        background: mode === id ? T.hover : "transparent",
        cursor: "pointer",
        marginRight: "4px",
      }}
    >
      {label}
    </button>
  );

  return (
    <Panel
      title="CONSTELLATION"
      sub={cls}
      accent={T.violet}
      right={
        <div style={{ display: "flex" }}>
          <Toggle id="split" label="SPLIT" />
          <Toggle id="raw" label="RAW" />
          <Toggle id="synced" label="SYNCED" />
        </div>
      }
    >
      {mode === "split" ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
          <div>
            <ConstellationCanvas points={raw} accent={T.red} />
            <div style={{ textAlign: "center", fontSize: "9px", marginTop: "4px", color: T.red, fontWeight: 600 }}>
              RAW / UNSYNCED
            </div>
          </div>
          <div>
            <ConstellationCanvas points={synced} accent={T.phosphor} />
            <div style={{ textAlign: "center", fontSize: "9px", marginTop: "4px", color: T.phosphor, fontWeight: 600 }}>
              STAGE 3 · SYNCHRONIZED
            </div>
          </div>
        </div>
      ) : (
        <div>
          <ConstellationCanvas points={mode === "raw" ? raw : synced} accent={mode === "raw" ? T.red : T.phosphor} />
          <div style={{ textAlign: "center", fontSize: "9px", marginTop: "4px", color: mode === "raw" ? T.red : T.phosphor, fontWeight: 600 }}>
            {mode === "raw" ? "RAW / UNSYNCED" : "STAGE 3 · SYNCHRONIZED"}
          </div>
        </div>
      )}
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// 6. Interleaver Panel
// ---------------------------------------------------------------------------
function InterleaverPanel({ interleaver }) {
  const { rank_profile, width, confidence, type } = interleaver;
  const dip = rank_profile.reduce((m, p) => (p.rank < m.rank ? p : m), rank_profile[0]);

  return (
    <Panel
      title="INTERLEAVER DETECTION"
      sub={`${type} · W=${width}`}
      accent={T.amber}
      right={<ConfidenceBar value={confidence} accent={T.amber} />}
    >
      <div style={{ height: 180, width: "100%" }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rank_profile} margin={{ top: 8, right: 10, left: -18, bottom: 0 }}>
            <CartesianGrid stroke={T.border} vertical={false} />
            <XAxis
              dataKey="width"
              tick={{ fill: T.textDim, fontSize: 9 }}
              stroke={T.border}
              label={{ value: "column width", position: "insideBottomRight", fill: T.textDim, fontSize: 9, offset: -2 }}
            />
            <YAxis
              tick={{ fill: T.textDim, fontSize: 9 }}
              stroke={T.border}
              width={34}
              label={{ value: "rank def.", angle: -90, position: "insideLeft", fill: T.textDim, fontSize: 9 }}
            />
            <Tooltip
              contentStyle={{ background: T.panel, border: `1px solid ${T.border}`, fontSize: 11 }}
              labelStyle={{ color: T.textDim }}
              itemStyle={{ color: T.amber }}
            />
            <ReferenceArea x1={dip.width - 0.5} x2={dip.width + 0.5} fill={T.amber} fillOpacity={0.12} />
            <Line type="monotone" dataKey="rank" stroke={T.amber} strokeWidth={1.5} dot={{ r: 2, fill: T.amber }} isAnimationActive={false} />
            <ReferenceDot x={dip.width} y={dip.rank} r={5} fill={T.panel} stroke={T.amber} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "4px", fontSize: "10px", color: T.textDim }}>
        <AlertTriangle size={11} color={T.amber} />
        <span>
          Minimum rank-deficiency at column width <span style={{ color: T.amber, fontWeight: 600 }}>{dip.width}</span> — declared
          interleaver width.
        </span>
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// 7. Hex Dump Viewer
// ---------------------------------------------------------------------------
const REGION_COLOR = {
  preamble: T.phosphor,
  header: T.cyan,
  payload: T.text,
  checksum: T.red,
};

function HexDumpPanel({ frames }) {
  const [hoveredRow, setHoveredRow] = useState(null);

  return (
    <Panel
      title="FRAME HEX DUMP"
      sub={`LEN ${frames.length}B · ${frames.preamble_matches} PREAMBLE MATCHES`}
      accent={T.text}
    >
      <div style={{ display: "flex", flexWrap: "wrap", gap: "12px", marginBottom: "8px" }}>
        {Object.entries(REGION_COLOR).map(([k, c]) => (
          <div key={k} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ width: "10px", height: "10px", background: c }} />
            <span style={{ fontSize: "9px", letterSpacing: "0.025em", color: T.textDim }}>
              {k.toUpperCase()}
            </span>
          </div>
        ))}
      </div>
      <div
        style={{
          maxHeight: 260,
          overflowY: "auto",
          border: `1px solid ${T.border}`,
          background: T.panel,
        }}
      >
        {frames.hex_dump.map((row, i) => (
          <div
            key={i}
            onMouseEnter={() => setHoveredRow(i)}
            onMouseLeave={() => setHoveredRow(null)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "12px",
              padding: "3px 8px",
              borderBottom: `1px solid ${T.border}`,
              background: hoveredRow === i ? T.hover : "transparent",
            }}
          >
            <span style={{ ...dataMono, fontSize: "10px", width: "56px", flexShrink: 0, color: T.textFaint }}>
              {row.offset.toString(16).padStart(4, "0").toUpperCase()}
            </span>
            <span
              style={{
                ...dataMono,
                fontSize: "11px",
                letterSpacing: "0.05em",
                flexShrink: 0,
                color: REGION_COLOR[row.region] || T.text,
                width: 200,
                fontWeight: 600,
              }}
            >
              {row.hex}
            </span>
            <span style={{ ...dataMono, fontSize: "10px", color: T.textDim }}>
              {row.ascii}
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// Root Dashboard (Full-width vertical stack matching your screenshot)
// ---------------------------------------------------------------------------
export default function SDRDashboard({ result, waterfallData }) {
  useGoogleFont();
  const [fileName, setFileName] = useState(null);
  const data = useMemo(() => result || buildMockResult(), [result]);
  const waterfall = useMemo(() => waterfallData || buildMockWaterfall(), [waterfallData]);
  const psd = useMemo(() => buildMockPSD(waterfall), [waterfall]);

  const handleFile = useCallback((file) => {
    setFileName(file.name);
  }, []);

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100%",
        backgroundColor: T.void,
        backgroundImage: T.bgGradient,
        color: T.text,
        ...mono,
        boxSizing: "border-box",
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 16px",
          background: T.panelHeader,
          borderBottom: `1px solid ${T.border}`,
          boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Radio size={16} color={T.phosphor} />
          <span style={{ fontSize: "13px", letterSpacing: "0.025em", fontWeight: 700, color: T.text }}>
            Blind Signal Exploitation Suite &mdash;
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "16px", fontSize: "10px", color: T.textDim }}>
          <span>FS {(data.fs / 1e6).toFixed(3)} MSPS</span>
          <span>RS {(data.rs / 1e3).toFixed(1)} KSPS</span>
          <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ width: "6px", height: "6px", background: T.phosphor, boxShadow: `0 0 6px ${T.phosphor}` }} />
            LIVE
          </span>
        </div>
      </header>

      {/* Stacked alignment matching your exact screenshot structure */}
      <main
        style={{
          padding: "16px",
          display: "flex",
          flexDirection: "column",
          gap: "14px",
          maxWidth: "100%",
          boxSizing: "border-box",
        }}
      >
        <UploadPanel fileName={fileName} onFile={handleFile} />
        <ExplainabilityCard modulation={data.modulation} fec={data.fec} />
        <WaterfallPanel waterfall={waterfall} fs={data.fs} />
        <PSDPanel psd={psd} rs={data.rs} />
        <ConstellationPanel cls={data.modulation.class} />
        <InterleaverPanel interleaver={data.interleaver} />
        <HexDumpPanel frames={data.frames} />
      </main>
    </div>
  );
}