/** Draggable resize handle for splitting analytics sections. */

import { useCallback, useEffect, useRef } from "react";

export function ResizeHandle({ onResize, ariaLabel = "Resize sections" }: { onResize: (deltaPx: number) => void; ariaLabel?: string }) {
  const draggingRef = useRef(false);
  const startYRef = useRef(0);

  const handleMouseDown = useCallback((event: React.MouseEvent) => {
    event.preventDefault();
    draggingRef.current = true;
    startYRef.current = event.clientY;
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
  }, []);

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      if (!draggingRef.current) return;
      const delta = event.clientY - startYRef.current;
      startYRef.current = event.clientY;
      onResize(delta);
    };
    const handleMouseUp = () => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [onResize]);

  return (
    <div
      role="separator"
      aria-orientation="horizontal"
      aria-label={ariaLabel}
      className="group relative h-1.5 shrink-0 cursor-row-resize bg-stone-800/60 hover:bg-cyan-300/40 active:bg-cyan-300/60"
      onMouseDown={handleMouseDown}
    >
      <div className="absolute left-1/2 top-1/2 h-0.5 w-8 -translate-x-1/2 -translate-y-1/2 rounded-full bg-stone-600 group-hover:bg-cyan-200/60" />
    </div>
  );
}