/** Component tests for the root App shell. */

import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { useAresStore } from "../state/useAresStore";

vi.mock("./GameViewport", () => ({
  GameViewport: () => <div data-testid="game-viewport">Mock Mars viewport</div>,
}));

describe("AresSim UI shell", () => {
  beforeEach(() => {
    useAresStore.getState().setRunMode("manual");
  });

  async function renderReadyApp() {
    render(<App />);
    await waitFor(() => expect(useAresStore.getState().snapshot).not.toBeNull());
    await waitFor(() => expect(useAresStore.getState().backendBusy).toBe(false));
  }

  it("renders the status ribbon and action shell", async () => {
    const user = userEvent.setup();
    await renderReadyApp();

    expect(await screen.findByText("ARESIM")).toBeInTheDocument();
    expect(screen.getByText(/SOL 001/)).toBeInTheDocument();
    expect(screen.getByTestId("mission-hud")).toBeInTheDocument();
    expect(screen.getByTestId("alert-hud")).toBeInTheDocument();
    expect(screen.queryByTestId("mission-drawer")).not.toBeInTheDocument();
    expect(screen.queryByTestId("agent-history")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Environment actions")).toBeInTheDocument();
    expect(screen.getByLabelText("Gameplay controls")).toBeInTheDocument();
    expect(screen.getByLabelText("Simulation mode")).toBeInTheDocument();
    expect(screen.getByLabelText("Export trajectory")).toBeInTheDocument();
    expect(screen.getByLabelText("World seed")).toBeInTheDocument();
    expect(screen.getByTestId("action-bar")).toHaveClass("h-[60px]");
    expect(screen.getByTestId("action-bar-left")).toContainElement(screen.getByLabelText("Simulation mode"));
    expect(screen.getByTestId("action-bar-center")).toContainElement(screen.getByLabelText("Environment actions"));
    expect(screen.getByTestId("action-bar-right")).toContainElement(screen.getByLabelText("Environment setup"));
    expect(screen.getByLabelText("Manual mode")).toHaveClass("text-cyan-200");
    expect(screen.getByLabelText("Scan").textContent).toBe("");
    expect(screen.getByTestId("header-rover-group")).toContainElement(screen.getByTestId("header-battery"));
    expect(screen.getByTestId("header-rover-group")).toContainElement(screen.getByTestId("header-health"));
    expect(screen.getByTestId("header-rover-group")).toContainElement(screen.getByTestId("header-storage"));
    expect(screen.getByTestId("header-build-pad-group")).toContainElement(screen.getByTestId("header-power"));
    expect(screen.getByTestId("header-build-pad-group")).toContainElement(screen.getByTestId("header-habitat"));
    expect(screen.getByTestId("header-build-pad-group")).toContainElement(screen.getByTestId("header-livability"));
    expect(screen.getByTestId("header-storage")).toHaveAttribute("aria-valuenow", "0");
    expect(screen.getByTestId("header-storage")).toHaveAttribute("aria-valuemax", "12");
    expect(screen.queryByTestId("payload-meter")).not.toBeInTheDocument();
    expect(screen.getByTestId("analytics-button")).toBeInTheDocument();
    expect(screen.getByTestId("header-power")).toHaveTextContent("Power");
    expect(screen.getByTestId("header-battery")).toHaveTextContent("Battery");
    expect(screen.getByTestId("header-health")).toHaveTextContent("Health");
    expect(screen.getByTestId("header-habitat")).toHaveTextContent("Habitat");
    expect(screen.getByTestId("header-livability")).toHaveTextContent("Livability");
    expect(within(screen.getByRole("banner")).getByText(/\+\d+\.\d{2} kW|-?\d+\.\d{2} kW/)).toBeInTheDocument();
    expect(screen.getAllByText(/\d+\.\d{2}%/).length).toBeGreaterThan(0);

    await user.click(screen.getByLabelText("Open mission"));
    expect(screen.getByTestId("mission-drawer")).toBeInTheDocument();
    expect(screen.getByText("Reward Objectives")).toBeInTheDocument();
    expect(screen.getByTestId("total-reward-card")).toBeInTheDocument();
    expect(screen.getByText("Ice Collected")).toBeInTheDocument();
    expect(screen.getByText("Payload Delivered")).toBeInTheDocument();
    expect(screen.getByText("Terrain Scanned")).toBeInTheDocument();
  });

  it("sets a deterministic custom seed from the command dock", async () => {
    const user = userEvent.setup();
    await renderReadyApp();

    await waitFor(() => expect(screen.getByLabelText("World seed")).toBeEnabled());
    await user.clear(screen.getByLabelText("World seed"));
    await user.type(screen.getByLabelText("World seed"), "43210");
    await user.click(screen.getByRole("button", { name: "Apply seed" }));

    await waitFor(() => expect(useAresStore.getState().snapshot?.seed).toBe(43210));
  });

  it("shows action tooltips and executes an action directly into history", async () => {
    const user = userEvent.setup();
    await renderReadyApp();

    await user.hover(screen.getByLabelText("Scan"));
    expect(await screen.findAllByText("Scan")).not.toHaveLength(0);
    await user.click(screen.getByLabelText("Scan"));
    await waitFor(() => expect(useAresStore.getState().snapshot?.step).toBe(1));

    await user.click(screen.getByLabelText("Open agent history"));
    const history = screen.getByTestId("agent-history");
    expect(within(history).getAllByText("scan").length).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: "Execute" })).not.toBeInTheDocument();
  });

  it("renders only the simplified visible action space", async () => {
    await renderReadyApp();

    expect(await screen.findByLabelText("Move")).toBeInTheDocument();
    expect(screen.getByLabelText("Scan")).toBeInTheDocument();
    expect(screen.getByLabelText("Extract")).toBeInTheDocument();
    expect(screen.getByLabelText("Build")).toBeInTheDocument();
    expect(screen.getByLabelText("Service")).toBeInTheDocument();
    expect(screen.getByLabelText("Unload payload")).toBeInTheDocument();
    expect(screen.getByLabelText("Wait")).toBeInTheDocument();
    expect(screen.queryByLabelText("Mine ice")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Mine ore")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Connect")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Repair")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Clean panel")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Invalid action test")).not.toBeInTheDocument();
  });

  it("switches mode configuration inside the unified command deck", async () => {
    const user = userEvent.setup();
    await renderReadyApp();

    await user.click(await screen.findByLabelText("Algorithm mode"));
    expect(screen.getByLabelText("Algorithm mode")).toHaveClass("text-violet-200");
    expect(screen.getByLabelText("Algorithm policy")).toBeInTheDocument();
    expect(screen.getByTestId("action-bar-left")).toContainElement(screen.getByLabelText("Algorithm policy"));
    expect(screen.getByTestId("action-bar-center")).toContainElement(screen.getByLabelText("Step simulation"));
    expect(screen.getByTestId("action-bar-right")).toContainElement(screen.getByLabelText("Randomize world"));
    expect(screen.queryByLabelText("Scan")).not.toBeInTheDocument();
    expect(screen.getByLabelText("World seed")).toBeInTheDocument();
    expect(screen.getByLabelText("Randomize world")).toBeInTheDocument();
    expect(screen.getByLabelText("Step simulation")).toBeInTheDocument();
    expect(screen.getByLabelText("Autoplay speed")).toBeInTheDocument();
    expect(screen.getByLabelText("Gameplay controls")).toBeInTheDocument();
    expect(screen.getByLabelText("Reset run")).toBeInTheDocument();
    const algorithmSeed = useAresStore.getState().snapshot?.seed;
    await user.click(screen.getByLabelText("Randomize world"));
    await waitFor(() => expect(useAresStore.getState().snapshot?.seed).not.toBe(algorithmSeed));
    expect(useAresStore.getState().runMode).toBe("algorithm");

    await user.click(screen.getByLabelText("Replay mode"));
    expect(screen.getByLabelText("Replay mode")).toHaveClass("text-amber-200");
    expect(screen.getByLabelText("Upload trajectory JSON")).toBeInTheDocument();
    expect(screen.getByTestId("loaded-replay-meta")).toHaveTextContent("No replay");
    expect(screen.getByLabelText("World seed")).toBeInTheDocument();
    expect(screen.getByLabelText("World seed")).not.toBeDisabled();
    expect(screen.getByLabelText("Randomize world")).toBeInTheDocument();
    expect(screen.getByLabelText("Randomize world")).not.toBeDisabled();
    expect(screen.queryByLabelText("Export trajectory")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Replay controls")).toBeInTheDocument();
    expect(screen.getByTestId("action-bar-left")).toContainElement(screen.getByLabelText("Upload trajectory JSON"));
    expect(screen.getByTestId("action-bar-center")).toContainElement(screen.getByLabelText("Replay controls"));
    expect(screen.getByLabelText("Play replay")).toBeDisabled();
    expect(screen.queryByLabelText("LLM mode")).not.toBeInTheDocument();
  });

  it("loads a saved trajectory-compatible JSON in load mode", async () => {
    const user = userEvent.setup();
    await renderReadyApp();

    const snapshot = useAresStore.getState().snapshot!;
    const file = new File([JSON.stringify({ savedAt: "now", snapshot })], "saved-run.json", { type: "application/json" });

    await user.click(screen.getByLabelText("Replay mode"));
    await waitFor(() => expect(useAresStore.getState().runMode).toBe("load"));
    await waitFor(() => expect(screen.getByLabelText("Upload trajectory JSON")).toBeInTheDocument());
    await user.upload(screen.getByLabelText("Upload trajectory JSON"), file);

    expect(await screen.findByRole("alert")).toHaveTextContent("Trajectory loaded");
    expect(screen.getByTestId("step-scrubber")).toBeInTheDocument();
    expect(screen.getByTestId("replay-cursor")).toBeInTheDocument();
    expect(useAresStore.getState().loadedReplay?.seed).toBe(snapshot.seed);
    expect(screen.getByLabelText("World seed")).toBeDisabled();
    expect(screen.getByLabelText("World seed")).toHaveValue(String(snapshot.seed));
  });

  it("filters agent history by action type from the dropdown", async () => {
    const user = userEvent.setup();
    await renderReadyApp();

    await user.click(await screen.findByLabelText("Scan"));
    await user.click(screen.getByLabelText("Open agent history"));
    await user.click(screen.getByRole("combobox", { name: "Filter agent history" }));
    await user.click(screen.getByRole("option", { name: "scan" }));

    expect(within(screen.getByTestId("agent-history")).getAllByText("scan").length).toBeGreaterThan(0);
  });

  it("opens the game guide modal", async () => {
    const user = userEvent.setup();
    await renderReadyApp();

    await user.click(await screen.findByLabelText("Open game guide"));

    expect(screen.getByText("AresSim Game Guide")).toBeInTheDocument();
    expect(screen.getByText("How To Play")).toBeInTheDocument();
    expect(screen.getByText("Controls")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "terrain" }));
    expect(screen.getByText("Environment Generation")).toBeInTheDocument();
    expect(screen.getByText("Terrain Legend")).toBeInTheDocument();
    expect(screen.getByText("Markers")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "actions" }));
    expect(screen.getByText("Validity Rules")).toBeInTheDocument();
    expect(screen.getByText("Action Space")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "rewards" }));
    expect(screen.getByText("How Rewards Are Computed")).toBeInTheDocument();
    expect(screen.getByText("Per-Action Reward Terms")).toBeInTheDocument();
    expect(screen.getByText("R = traversal − (battery cost × 0.08) + safety")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "rules" }));
    expect(screen.getByText("Terminal Failure Conditions")).toBeInTheDocument();
    expect(screen.getByText("Battery Drain Formula")).toBeInTheDocument();
    expect(screen.getByText(/D = base\(action\)/)).toBeInTheDocument();
    expect(screen.queryByText("Mission Tasks")).not.toBeInTheDocument();
  });

  it("renders habitat build progress in the contextual mission drawer", async () => {
    const user = userEvent.setup();
    await renderReadyApp();
    const snapshot = useAresStore.getState().snapshot!;
    const pad = snapshot.terrain.flat().find((cell) => cell.terrain === "build_pad")!;

    for (let index = 0; index < 10; index += 1) {
      await act(async () => {
        await useAresStore.getState().dispatchAction({ type: "build", target: { x: pad.x, y: pad.y } });
      });
    }

    await waitFor(() => expect(useAresStore.getState().snapshot?.objectiveStats.habitatBuildCount).toBe(10));
    await user.click(screen.getByLabelText("Open mission"));
    expect(screen.getByText(/100\.00% \(10\/10\)/)).toBeInTheDocument();
  });

  it("does not expose buildability in the inspector UI", async () => {
    await renderReadyApp();
    act(() => {
      useAresStore.getState().selectTarget({ kind: "cell", x: 12, y: 12 });
    });

    expect(screen.queryByText("Buildability")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "buildability" })).not.toBeInTheDocument();
  });

  it("shows a visible warning for invalid environment actions", async () => {
    await renderReadyApp();
    await act(async () => useAresStore.getState().dispatchAction({ type: "build", target: { x: 0, y: 0 } }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Blocked action");
  });

  it("opens the analytics modal from the ribbon button and the total reward card", async () => {
    const user = userEvent.setup();
    await renderReadyApp();
    await user.click(screen.getByTestId("analytics-button"));
    expect(await screen.findByText("Run Analytics")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Rewards/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Resources/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Behavior/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Progress/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Environment/ })).toBeInTheDocument();
  });
});
