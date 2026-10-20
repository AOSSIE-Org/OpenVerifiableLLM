"use client"

import createGlobe, { COBEOptions } from "cobe"
import { useCallback, useEffect, useRef } from "react"

import { cn } from "@/lib/utils"

const GLOBE_CONFIG: COBEOptions = {
  width: 800,
  height: 800,
  onRender: () => {},
  devicePixelRatio: 2,
  phi: 0,
  theta: 0.3,
  dark: 0,
  diffuse: 0.9,
  mapSamples: 16000,
  mapBrightness: 1.4,
  baseColor: [0.84, 0.88, 0.89],
  markerColor: [94 / 255, 108 / 255, 91 / 255],
  glowColor: [0.2, 0.3, 0.3],
  markers: [
    { location: [14.5995, 120.9842], size: 0.03 },
    { location: [19.076, 72.8777], size: 0.1 },
    { location: [23.8103, 90.4125], size: 0.05 },
    { location: [30.0444, 31.2357], size: 0.07 },
    { location: [39.9042, 116.4074], size: 0.08 },
    { location: [-23.5505, -46.6333], size: 0.1 },
    { location: [19.4326, -99.1332], size: 0.1 },
    { location: [40.7128, -74.006], size: 0.1 },
    { location: [34.6937, 135.5022], size: 0.05 },
    { location: [41.0082, 28.9784], size: 0.06 },
  ],
}

export function Globe({
  className,
  config = GLOBE_CONFIG,
}: {
  className?: string
  config?: COBEOptions
}) {
  const phi = useRef(0)
  const width = useRef(0)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const pointerInteracting = useRef<number | null>(null)
  const pointerInteractionMovement = useRef(0)
  const r = useRef(0)

  const updatePointerInteraction = (value: number | null) => {
    pointerInteracting.current = value
    if (canvasRef.current) {
      canvasRef.current.style.cursor = value !== null ? "grabbing" : "grab"
    }
  }

  const updateMovement = (clientX: number) => {
    if (pointerInteracting.current !== null) {
      const delta = clientX - pointerInteracting.current
      pointerInteractionMovement.current = delta
      r.current = delta / 200
    }
  }

  const onRender = useCallback(
    (state: Record<string, any>) => {
      if (pointerInteracting.current === null) phi.current += 0.005
      state.phi = phi.current + r.current
      state.width = width.current * 2
      state.height = width.current * 2
    },
    [],
  )

  const onResize = () => {
    if (canvasRef.current) {
      width.current = canvasRef.current.offsetWidth
    }
  }

  useEffect(() => {
    window.addEventListener("resize", onResize)
    onResize()

    let globe: ReturnType<typeof createGlobe> | undefined
    try {
      globe = createGlobe(canvasRef.current!, {
        ...config,
        width: width.current * 2,
        height: width.current * 2,
        onRender,
      })
    } catch {
      // WebGL unavailable: the footer still works without the globe.
    }

    const timer = setTimeout(() => {
      if (canvasRef.current) canvasRef.current.style.opacity = "1"
    })
    return () => {
      clearTimeout(timer)
      window.removeEventListener("resize", onResize)
      globe?.destroy()
    }
  }, [])

  return (
    <div
      aria-hidden="true"
      className={cn(
        "absolute inset-0 mx-auto aspect-[1/1] w-full max-w-[600px]",
        className,
      )}
    >
      <canvas
        className={cn(
          "size-full opacity-0 transition-opacity duration-500 [contain:layout_paint_size]",
        )}
        ref={canvasRef}
        onPointerDown={(e) =>
          updatePointerInteraction(e.clientX - pointerInteractionMovement.current)
        }
        onPointerUp={() => updatePointerInteraction(null)}
        onPointerOut={() => updatePointerInteraction(null)}
        onMouseMove={(e) => updateMovement(e.clientX)}
        onTouchMove={(e) => e.touches[0] && updateMovement(e.touches[0].clientX)}
      />
    </div>
  )
}
