import { useState, useEffect, useRef, useCallback } from 'react';

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface WebSocketMessage {
    type: string;
    [key: string]: unknown;
}

export interface UseWebSocketOptions {
    url: string;
    autoConnect?: boolean;
    reconnectInterval?: number;
    maxReconnectAttempts?: number;
    onMessage?: (data: WebSocketMessage) => void;
    onConnect?: () => void;
    onDisconnect?: () => void;
    onError?: (error: Event) => void;
}

export interface UseWebSocketReturn {
    status: ConnectionStatus;
    lastMessage: WebSocketMessage | null;
    connect: () => void;
    disconnect: () => void;
    send: (data: WebSocketMessage) => boolean;
    isConnected: boolean;
}

export const useWebSocket = (options: UseWebSocketOptions): UseWebSocketReturn => {
    const {
        url,
        autoConnect = false,
        reconnectInterval = 3000,
        maxReconnectAttempts = 5,
        onMessage,
        onConnect,
        onDisconnect,
        onError,
    } = options;

    const [status, setStatus] = useState<ConnectionStatus>('disconnected');
    const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectAttempts = useRef(0);
    const reconnectTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
    const isManualDisconnect = useRef(false);

    // Store callbacks in refs to prevent reconnection on callback changes
    const onMessageRef = useRef(onMessage);
    const onConnectRef = useRef(onConnect);
    const onDisconnectRef = useRef(onDisconnect);
    const onErrorRef = useRef(onError);

    // Update refs when callbacks change
    useEffect(() => { onMessageRef.current = onMessage; }, [onMessage]);
    useEffect(() => { onConnectRef.current = onConnect; }, [onConnect]);
    useEffect(() => { onDisconnectRef.current = onDisconnect; }, [onDisconnect]);
    useEffect(() => { onErrorRef.current = onError; }, [onError]);

    const clearReconnectTimeout = useCallback(() => {
        if (reconnectTimeout.current) {
            clearTimeout(reconnectTimeout.current);
            reconnectTimeout.current = null;
        }
    }, []);

    const disconnect = useCallback(() => {
        clearReconnectTimeout();
        isManualDisconnect.current = true;
        reconnectAttempts.current = maxReconnectAttempts; // Prevent auto-reconnect

        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setStatus('disconnected');
    }, [clearReconnectTimeout, maxReconnectAttempts]);

    const connect = useCallback(() => {
        // Don't connect if already connected or connecting
        if (wsRef.current?.readyState === WebSocket.OPEN ||
            wsRef.current?.readyState === WebSocket.CONNECTING) {
            return;
        }

        clearReconnectTimeout();
        isManualDisconnect.current = false;
        setStatus('connecting');

        try {
            const ws = new WebSocket(url);
            wsRef.current = ws;

            ws.onopen = () => {
                setStatus('connected');
                reconnectAttempts.current = 0;
                onConnectRef.current?.();
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data) as WebSocketMessage;
                    setLastMessage(data);
                    onMessageRef.current?.(data);
                } catch (err) {
                    console.error('Failed to parse WebSocket message:', err);
                }
            };

            ws.onclose = () => {
                wsRef.current = null;
                setStatus('disconnected');
                onDisconnectRef.current?.();

                // Attempt reconnect if not manually disconnected and haven't exceeded max attempts
                if (!isManualDisconnect.current && reconnectAttempts.current < maxReconnectAttempts) {
                    reconnectAttempts.current++;
                    reconnectTimeout.current = setTimeout(() => {
                        connect();
                    }, reconnectInterval);
                }
            };

            ws.onerror = (error) => {
                setStatus('error');
                onErrorRef.current?.(error);
            };

        } catch (err) {
            console.error('Failed to create WebSocket:', err);
            setStatus('error');
        }
    }, [url, clearReconnectTimeout, reconnectInterval, maxReconnectAttempts]);

    const send = useCallback((data: WebSocketMessage): boolean => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(data));
            return true;
        }
        return false;
    }, []);

    // Auto-connect on mount if enabled
    useEffect(() => {
        if (autoConnect) {
            connect();
        }

        return () => {
            isManualDisconnect.current = true;
            reconnectAttempts.current = maxReconnectAttempts; // Prevent reconnect on unmount
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
            clearReconnectTimeout();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [autoConnect]); // Only run on mount/unmount and autoConnect change

    return {
        status,
        lastMessage,
        connect,
        disconnect,
        send,
        isConnected: status === 'connected',
    };
};

