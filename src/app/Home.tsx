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
  const raw = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8080";
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
  const [detections, setDetections] = useState<Detection[]>([])

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
            setDetections(data.detections);
            
            const uniqueLabels = Array.from(new Set(data.detections.map(d => d.label)));
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
  const target = detections.length > 0 ? detections[0] : null;

  return (
    <main className="min-h-screen w-full text-gray-100 flex p-4 pt-20 gap-4 bg-[#ffffff]">
      <div className="flex-[2] flex flex-col">
        <div className="relative h-full rounded-2xl overflow-hidden border border-gray-200 bg-black shadow-sm">
          <img
            ref={imgRef}
            src={STREAM_URL}
            alt="Live camera feed"
            className="w-full h-full object-contain"
          />
        </div>
      </div>

      <div className="flex-1 flex flex-col h-full justify-center py-12">
        <div className="flex flex-col gap-6 h-full max-h-[90%]">
          
            <div className={`p-8 flex flex-col justify-center items-center flex-1 rounded-3xl shadow-xl w-full transition-colors duration-300 ${target ? 'bg-emerald-600' : 'bg-emerald-600/90'}`}>
              <span className={`text-xl font-bold uppercase tracking-widest block mb-6 ${target ? 'text-emerald-100' : 'text-white/70'}`}>
                Location
              </span>
              <div className={`text-8xl xl:text-9xl font-black leading-tight break-words text-center ${target ? 'text-white' : 'text-white/50'}`}>
                {target ? target.label : "-"}
              </div>
            </div>
            
            <div className={`p-8 flex flex-col justify-center items-center flex-1 rounded-3xl shadow-xl w-full transition-colors duration-300 ${target ? 'bg-[#005496]' : 'bg-[#005496]/90'}`}>
               <span className={`text-xl font-bold uppercase tracking-widest block mb-6 ${target ? 'text-blue-100' : 'text-white/70'}`}>
                  Accuracy Rate
               </span>
               <div className={`text-8xl xl:text-9xl font-black leading-tight break-words text-center ${target ? 'text-white' : 'text-white/50'}`}>
                  {/* {target ? `${(target.conf * 100).toFixed(1)}%` : "-"} */}
                  {target ? target.conf.toFixed(2) : "-"}
               </div>
            </div>

        </div>
      </div>
    </main>
  );
}