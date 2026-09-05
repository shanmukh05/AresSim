import "@testing-library/jest-dom/vitest";
import { beforeEach } from "vitest";
import { installMockApi, resetMockApi } from "./src/test/mockApi";

installMockApi();
beforeEach(() => resetMockApi());

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

(globalThis as unknown as { ResizeObserver: typeof ResizeObserverMock }).ResizeObserver = ResizeObserverMock;

const htmlElementPrototype = (globalThis as typeof globalThis & { HTMLElement?: { prototype: Record<string, unknown> } }).HTMLElement?.prototype;

if (htmlElementPrototype) {
  htmlElementPrototype.hasPointerCapture ??= () => false;
  htmlElementPrototype.setPointerCapture ??= () => {};
  htmlElementPrototype.releasePointerCapture ??= () => {};
  htmlElementPrototype.scrollIntoView ??= () => {};
}
