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

export default function HomePage() {
  // const API_BASE = useMemo(resolveApiBase, []);
  const [apiBase, setApiBase] = useState<string | null>(null)
  const STREAM_URL = apiBase ? `${apiBase}/stream.mjpg`: undefined
  const WS_URL = useMemo(() => {
    if (!apiBase) return ""
    try {
      const u = new URL(apiBase)
      u.protocol = u.protocol === "https:" ? "wss:" : "ws:"
      u.pathname = "/ws"
      return u.toString()
    } catch {
      return ""
    }
  }, [apiBase])
  
  const imgRef = useRef<HTMLImageElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [now, setNow] = useState<Date | null>(null);
  const [labels, setLabels] = useState<string[]>([]);
  const [detections, setDetections] = useState<Detection[]>([])

  useEffect(() => {
    fetch("/config.json")
      .then((res) => res.json())
      .then((config) => {
        const url = config.apiBase
        setApiBase(url)
      })
      .catch((err) => console.error("Error loading config:", err))
  }, [])

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
    if (!WS_URL) return

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