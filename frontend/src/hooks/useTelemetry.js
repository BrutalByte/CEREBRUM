import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = `ws://${window.location.host}/ws/telemetry`
const MAX_EVENTS = 200

export function useTelemetry() {
  const [events, setEvents] = useState([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const reconnectRef = useRef(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      if (reconnectRef.current) {
        clearTimeout(reconnectRef.current)
        reconnectRef.current = null
      }
    }

    ws.onclose = () => {
      setConnected(false)
      reconnectRef.current = setTimeout(connect, 3000)
    }

    ws.onerror = () => {
      ws.close()
    }

    ws.onmessage = (msg) => {
      try {
        const e = JSON.parse(msg.data)
        setEvents(prev => [{ ...e, _id: Date.now() + Math.random() }, ...prev].slice(0, MAX_EVENTS))
      } catch (_) {}
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectRef.current) clearTimeout(reconnectRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  const latestByType = (type) => events.find(e => e.event_type === type)

  return { events, connected, latestByType }
}
