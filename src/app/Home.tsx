"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Box = { x1: number; y1: number; x2: number; y2: number };
type Detection = { cls: number; label: string; conf: number; box: Box };
type WsPayload = {
  ts: number;
  width: number;
  height: number;
  detections: Detection[];
  latency_ms?: number;
  error?: string;
};

function resolveApiBase(): string {
  const raw = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
  return raw.replace(/\/+$/, "");
}

// use client camera but must have HTTPS
function toWsClientUrl(httpBase: string): string {
  try {
    const u = new URL(httpBase);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    u.pathname = "/ws-client";
    u.search = "";
    return u.toString();
  } catch {
    return "ws://localhost:8000/ws-client";
  }
}

// server camera
function toWsUrl(httpBase: string): string {
  try {
    const u = new URL(httpBase);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    u.pathname = "/ws";
    u.search = "";
    return u.toString();
  } catch {
    return "ws://localhost:8000/ws";
  }
}

function drawDetections(
  canvas: HTMLCanvasElement,
  payload: WsPayload,
  displayW: number,
  displayH: number
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = Math.max(1, window.devicePixelRatio || 1);

  const targetW = Math.round(displayW * dpr);
  const targetH = Math.round(displayH * dpr);
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
  canvas.style.width = `${displayW}px`;
  canvas.style.height = `${displayH}px`;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(dpr, dpr);

  const sx = displayW / payload.width;
  const sy = displayH / payload.height;

  ctx.lineWidth = 2;
  ctx.font = "12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto";
  payload.detections.forEach((d) => {
    const x1 = d.box.x1 * sx;
    const y1 = d.box.y1 * sy;
    const x2 = d.box.x2 * sx;
    const y2 = d.box.y2 * sy;
    const w = x2 - x1;
    const h = y2 - y1;

    ctx.strokeStyle = "rgba(16,185,129,1)";
    ctx.strokeRect(x1, y1, w, h);

    const label = `${d.label} ${d.conf.toFixed(2)}`;
    const padX = 4;
    const padY = 2;
    const textW = ctx.measureText(label).width;
    const textH = 12;
    ctx.fillStyle = "rgba(16,185,129,1)";
    ctx.fillRect(
      x1,
      Math.max(0, y1 - (textH + padY * 2)),
      textW + padX * 2,
      textH + padY * 2
    );

    ctx.fillStyle = "rgba(0,0,0,1)";
    ctx.fillText(label, x1 + padX, Math.max(textH + padY, y1 - padY));
  });

  ctx.restore();
}

export default function HomePage() {
  const API_BASE = useMemo(resolveApiBase, []);
  const STREAM_URL = `${API_BASE}/stream.mjpg`;
  const WS_URL = useMemo(() => {
    const u = new URL(API_BASE);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    u.pathname = "/ws";
    return u.toString();
  }, [API_BASE]);
  
  const imgRef = useRef<HTMLImageElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [now, setNow] = useState<Date | null>(null);
  const [labels, setLabels] = useState<string[]>([]);

  // Clock
  useEffect(() => {
    setNow(new Date());
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  // Check stream connection
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;

    const handleLoad = () => {
      setIsConnected(true);
      setIsLoading(false);
      setError(null);
    };

    const handleError = () => {
      setIsConnected(false);
      setIsLoading(false);
      setError("Cannot connect to camera stream");
    };

    img.addEventListener("load", handleLoad);
    img.addEventListener("error", handleError);

    return () => {
      img.removeEventListener("load", handleLoad);
      img.removeEventListener("error", handleError);
    };
  }, []);

  // WebSocket for detections data
  useEffect(() => {
    let alive = true;
    let reconnectTimeout: NodeJS.Timeout;

    const connectWs = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("✅ WebSocket connected");
      };

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data) as WsPayload;
          if (!("error" in data) && data.detections) {
            // Extract unique labels
            const uniqueLabels = Array.from(
              new Set(data.detections.map(d => d.label))
            );
            setLabels(uniqueLabels);
          }
        } catch (e) {
          console.error("Failed to parse WS message:", e);
        }
      };

      ws.onerror = () => {
        console.error("WebSocket error");
      };

      ws.onclose = (ev) => {
        if (alive && ev.code !== 1000) {
          reconnectTimeout = setTimeout(() => {
            if (alive) connectWs();
          }, 3000);
        }
      };
    };

    connectWs();

    return () => {
      alive = false;
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      wsRef.current?.close(1000);
    };
  }, [WS_URL]);

  const dateStr = useMemo(() => {
    if (!now) return null;
    return new Intl.DateTimeFormat("en-US", {
      timeZone: "Asia/Bangkok",
      month: "2-digit",
      day: "2-digit",
      year: "numeric",
    }).format(now);
  }, [now]);

  const timeStr = useMemo(() => {
    if (!now) return null;
    return new Intl.DateTimeFormat("en-US", {
      timeZone: "Asia/Bangkok",
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }).format(now);
  }, [now]);

  const dateTimeStr = dateStr && timeStr ? `${dateStr}●${timeStr}` : null;

  return (
    <main className="min-h-screen w-full text-gray-100 flex items-center justify-center p-4">
      <div className="w-full max-w-3xl">
        {/* {isLoading && (
          <div className="mb-4 p-4 bg-blue-500/20 border border-blue-500/40 rounded-lg text-blue-200">
            <div className="flex items-center gap-2">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
              </svg>
              Connecting to camera...
            </div>
          </div>
        )} */}

        {error && (
          <div className="mb-4 p-4 bg-red-500/20 border border-red-500/40 rounded-lg text-red-200">
            {error}
          </div>
        )}

        <div className="relative mx-auto w-full">
          {/* Label bar */}
          <div className="absolute left-0 bottom-full mb-3 z-20">
            <div className="rounded-full bg-gray-900/70 border border-gray-800 px-3 py-2 backdrop-blur">
              {labels.length ? (
                <div className="flex flex-wrap gap-2 items-center">
                  {labels.map((l, i) => (
                    <span
                      key={`${l}-${i}`}
                      className="text-sm font-medium px-2.5 py-1 rounded-full bg-purple-500/20 border border-purple-500/40 text-purple-200"
                    >
                      Location: {l}
                    </span>
                  ))}
                </div>
              ) : (
                <span className="text-sm text-gray-400">No detections</span>
              )}
            </div>
          </div>

          {/* LIVE + Time */}
          <div className="absolute right-0 bottom-full mb-3 z-20">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium px-2.5 py-1 rounded-full bg-gray-900/70 border border-gray-800 text-gray-200 backdrop-blur inline-flex items-center gap-1">
                <svg aria-hidden viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="4" width="18" height="18" rx="2" />
                  <line x1="16" y1="2" x2="16" y2="6" />
                  <line x1="8" y1="2" x2="8" y2="6" />
                  <line x1="3" y1="10" x2="21" y2="10" />
                </svg>
                {dateTimeStr ?? "--/--/----●--:--:--"}
              </span>

              {/* <span className={`text-sm font-semibold px-2.5 py-1 rounded-full ${
                isConnected 
                  ? "bg-[#CCF1C8] border-[#548664] text-[#36BC55]"
                  : "bg-gray-500/20 border-gray-500/40 text-gray-400"
              } border`}>
                ● {isConnected ? "LIVE" : "CONNECTING"}
              </span> */}
            </div>
          </div>

          {/* Camera stream */}
          <div className="relative rounded-2xl overflow-hidden shadow-2xl border border-gray-800 bg-black w-full">
            <img
              ref={imgRef}
              src={STREAM_URL}
              alt="Live camera feed"
              className="block w-full h-auto object-contain"
            />
          </div>
        </div>
      </div>
    </main>
  );
}