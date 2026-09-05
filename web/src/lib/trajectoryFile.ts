/** Browser file transport for backend-owned trajectory JSON. */

/** Download a trajectory payload without interpreting or rebuilding its schema. */
export function downloadTrajectoryFile(fileName: string, payload: unknown) {
  if (typeof window === "undefined" || typeof document === "undefined" || typeof URL === "undefined" || typeof URL.createObjectURL !== "function") return;
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName.endsWith(".json") ? fileName : `${fileName}.json`;
  link.click();
  URL.revokeObjectURL(url);
}
