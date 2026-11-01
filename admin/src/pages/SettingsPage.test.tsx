import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { setAdminTransport } from "../api/client";
import type { MockBackend } from "../mock/backend";
import { startMockBackend } from "../testing/harness";
import { SettingsPage } from "./SettingsPage";

let backend: MockBackend;
beforeEach(() => {
  backend = startMockBackend();
});

const saved = (name: string) => backend.state.settings[name];

async function open() {
  render(<SettingsPage />);
  await screen.findByLabelText("Confidence threshold");
  return userEvent.setup();
}

async function type(user: ReturnType<typeof userEvent.setup>, label: string, value: string) {
  const input = screen.getByLabelText(label);
  await user.clear(input);
  await user.type(input, value);
}

describe("SettingsPage", () => {
  it("builds the form from GET /settings: type, range, default and who last changed it", async () => {
    await open();

    const threshold = screen.getByLabelText("Confidence threshold");
    expect(threshold).toHaveValue(0.45);
    expect(threshold).toHaveAttribute("min", "0.25");
    expect(threshold).toHaveAttribute("max", "0.8");
    expect(screen.getByLabelText("Gemini verify")).toBeChecked();
    expect(screen.getByLabelText("Worker auth")).toHaveValue("optional");
    expect(screen.getByLabelText("Gemini min confidence")).toHaveValue(0.8); // saved at 0.8 by someone else
    expect(screen.getByText(/Last changed by second\.admin@example\.com/)).toBeInTheDocument();
  });

  it("keeps Save off until something changes", async () => {
    const user = await open();
    expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();

    await type(user, "Gemini min confidence", "0.9");
    expect(screen.getByRole("button", { name: "Save changes" })).toBeEnabled();

    await type(user, "Gemini min confidence", "0.8"); // back to what it was
    expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
  });

  it("saves a setting that needs no confirmation straight away, and says when it applies", async () => {
    const user = await open();
    await type(user, "Gemini min confidence", "0.9");

    await user.click(screen.getByRole("button", { name: "Save changes" }));

    expect(await screen.findByText(/Saved\. It takes effect within 30 seconds/)).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(saved("gemini_min_confidence").value).toBe(0.9);
  });

  describe("a setting that asks for confirmation (the detector threshold, worker login)", () => {
    it("shows the old and new value first; Cancel saves nothing", async () => {
      const user = await open();
      await type(user, "Confidence threshold", "0.6");
      await user.click(screen.getByRole("button", { name: "Save changes" }));

      const dialog = await screen.findByRole("dialog", { name: "Confirm these changes" });
      expect(dialog).toHaveTextContent("Confidence threshold: 0.45 → 0.6");
      await user.click(within(dialog).getByRole("button", { name: "Cancel" }));

      expect(screen.queryByRole("dialog")).toBeNull();
      expect(saved("confidence_threshold")).toBeUndefined();
      expect(screen.getByLabelText("Confidence threshold")).toHaveValue(0.6); // the edit is still there
    });

    it("saves on 'Yes, save' and shows the warning", async () => {
      const user = await open();
      await type(user, "Confidence threshold", "0.6");
      await user.click(screen.getByRole("button", { name: "Save changes" }));

      await user.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Yes, save" }));

      expect(await screen.findByText(/Every verdict from now on uses the new detector threshold/)).toBeInTheDocument();
      expect(saved("confidence_threshold").value).toBe(0.6);
    });

    it("warns that checks from an app without login will be refused", async () => {
      const user = await open();
      await user.selectOptions(screen.getByLabelText("Worker auth"), "required");
      await user.click(screen.getByRole("button", { name: "Save changes" }));
      const dialog = await screen.findByRole("dialog");
      expect(dialog).toHaveTextContent("Worker auth: optional → required");

      await user.click(within(dialog).getByRole("button", { name: "Yes, save" }));

      expect(await screen.findByText(/Checks from an app without login will now be refused/)).toBeInTheDocument();
    });
  });

  it("warns when Gemini verification is switched off", async () => {
    const user = await open();
    await user.click(screen.getByLabelText("Gemini verify"));

    await user.click(screen.getByRole("button", { name: "Save changes" }));

    expect(await screen.findByText(/Gemini verification is off/)).toBeInTheDocument();
    expect(saved("gemini_verify").value).toBe(false);
  });

  it("'Use default' sends null, so the setting goes back to the server's default", async () => {
    const user = await open();

    await user.click(screen.getByRole("button", { name: "Use the default for Gemini min confidence" }));
    expect(screen.getByText("Will go back to the default (0.75).")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Save changes" }));

    expect(await screen.findByText(/Saved\./)).toBeInTheDocument();
    expect(saved("gemini_min_confidence")).toBeUndefined();
    expect(screen.getByLabelText("Gemini min confidence")).toHaveValue(0.75);
  });

  describe("a number outside the range the setting describes", () => {
    it("is marked as soon as it's typed: no confirmation, no request, Save off", async () => {
      const asked: string[] = [];
      const real = backend.fetch;
      setAdminTransport(async (url, init) => {
        asked.push(`${init?.method ?? "GET"} ${url}`);
        return real(url, init);
      });
      const user = await open();

      await type(user, "Confidence threshold", "0.9"); // the range shown is 0.25 to 0.8

      expect(await screen.findByText("Must be between 0.25 and 0.8.")).toBeInTheDocument();
      expect(screen.getByLabelText("Confidence threshold")).toHaveAttribute("aria-invalid", "true");
      expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
      expect(screen.queryByRole("dialog")).toBeNull();
      expect(asked.filter((a) => a.startsWith("PATCH"))).toEqual([]);
    });

    it("also stops a value below the range, and clears the mark when it's fixed", async () => {
      const user = await open();
      await type(user, "Confidence threshold", "0.1");
      expect(await screen.findByText("Must be between 0.25 and 0.8.")).toBeInTheDocument();

      await type(user, "Confidence threshold", "0.5");

      expect(screen.queryByText("Must be between 0.25 and 0.8.")).toBeNull();
      expect(screen.getByRole("button", { name: "Save changes" })).toBeEnabled();
    });

    it("accepts the two ends of the range", async () => {
      const user = await open();
      for (const edge of ["0.25", "0.8"]) {
        await type(user, "Confidence threshold", edge);
        expect(screen.queryByText(/Must be between/)).toBeNull();
        expect(screen.getByRole("button", { name: "Save changes" })).toBeEnabled();
      }
    });

    it("doesn't block a value that is between the ends but off the step: only min and max are rules", async () => {
      const user = await open();

      await type(user, "Confidence threshold", "0.47"); // step is 0.05

      expect(screen.queryByText(/Must be between/)).toBeNull();
      expect(screen.getByRole("button", { name: "Save changes" })).toBeEnabled();
    });

    it("is still all or nothing when the server refuses: a page that showed an older, wider range", async () => {
      // The descriptor the page loaded says 0.25–1.0, as if an admin narrowed it after the page loaded.
      const real = backend.fetch;
      setAdminTransport(async (url, init) => {
        const response = await real(url, init);
        if (url.endsWith("/admin/settings") && (init?.method ?? "GET") === "GET") {
          const list = await response.json();
          list.find((s: { name: string }) => s.name === "confidence_threshold").max = 1;
          return new Response(JSON.stringify(list), { status: 200, headers: { "Content-Type": "application/json" } });
        }
        return response;
      });
      const user = await open();
      await user.click(screen.getByLabelText("Gemini verify")); // fine on its own
      await type(user, "Confidence threshold", "0.95"); // allowed by the page's range, not by the server's
      await user.click(screen.getByRole("button", { name: "Save changes" }));
      await user.click(within(await screen.findByRole("dialog")).getByRole("button", { name: "Yes, save" }));

      expect(await screen.findByText(/Nothing was saved/)).toBeInTheDocument();
      expect(screen.getByText(/confidence_threshold must be between 0\.25 and 0\.8/)).toBeInTheDocument();
      expect(screen.getByLabelText("Confidence threshold")).toHaveAttribute("aria-invalid", "true");
      expect(saved("gemini_verify")).toBeUndefined(); // the valid change wasn't saved either
    });
  });

  it("can't save a number that has been cleared", async () => {
    const user = await open();

    await user.clear(screen.getByLabelText("Confidence threshold"));

    expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
  });

  it("discards everything not yet saved", async () => {
    const user = await open();
    await type(user, "Gemini min confidence", "0.9");

    await user.click(screen.getByRole("button", { name: "Discard changes" }));

    expect(screen.getByLabelText("Gemini min confidence")).toHaveValue(0.8);
  });
});
