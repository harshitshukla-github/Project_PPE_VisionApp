import { fireEvent, render, screen } from "@testing-library/react-native";

import CheckResultView from "../CheckResultView";
import type { ComplianceCheckResponse } from "../../api/types";

const FAIL_RESULT: ComplianceCheckResponse = {
  worker_id: "W-0042",
  job_category: "welder",
  timestamp: "2026-07-29T13:08:34Z",
  verdict: "FAIL",
  confidence_avg: 0.7,
  detections: [
    { class: "hardhat", confidence: 0.9, bbox: [100, 200, 300, 600] },
    { class: "no_hardhat", confidence: 0.6, bbox: [0, 0, 50, 50] },
    { class: "person", confidence: 0.99, bbox: [0, 0, 1000, 2000] },
  ],
  missing_equipment: [],
  violations_detected: ["no_hardhat"],
  unverifiable_equipment: [],
  explanation: "You're missing your hardhat.",
  model_used: "yolo26n",
  inference_time_ms: 30.1,
  gemini_invoked: true,
};

/** Fires the container's onLayout then the image's onLoad to resolve the bbox scale/offset. */
function measurePhoto(containerWidth: number, naturalWidth: number, naturalHeight: number) {
  fireEvent(screen.getByTestId("check-result-photo-container"), "layout", {
    nativeEvent: { layout: { x: 0, y: 0, width: containerWidth, height: 360 } },
  });
  fireEvent(screen.getByTestId("check-result-photo"), "load", {
    nativeEvent: { source: { width: naturalWidth, height: naturalHeight, uri: "file:///photo.jpg" } },
  });
}

describe("CheckResultView bbox overlay", () => {
  it("scales detection boxes from the image's natural size to the rendered container", () => {
    render(<CheckResultView result={FAIL_RESULT} imageUri="file:///photo.jpg" />);

    // naturalSize 1000x2000 into a 300x360 "contain" box:
    // scale = min(300/1000, 360/2000) = 0.18, offsetX = 60, offsetY = 0
    measurePhoto(300, 1000, 2000);

    const hardhatBox = screen.getByTestId("bbox-hardhat-0");
    expect(hardhatBox).toHaveStyle({ left: 78, top: 36, width: 36, height: 72 });
  });

  it("colors a compliant item green, a no_* violation red, and person neutral gray", () => {
    render(<CheckResultView result={FAIL_RESULT} imageUri="file:///photo.jpg" />);
    measurePhoto(300, 1000, 2000);

    expect(screen.getByTestId("bbox-hardhat-0")).toHaveStyle({ borderColor: "#1a7f37" });
    expect(screen.getByTestId("bbox-no_hardhat-1")).toHaveStyle({ borderColor: "#c0392b" });
    expect(screen.getByTestId("bbox-person-2")).toHaveStyle({ borderColor: "#607d8b" });
  });

  it("labels each box with the title-cased class and confidence percentage", () => {
    render(<CheckResultView result={FAIL_RESULT} imageUri="file:///photo.jpg" />);
    measurePhoto(300, 1000, 2000);

    expect(screen.getByText("Hardhat 90%")).toBeTruthy();
    expect(screen.getByText("No Hardhat 60%")).toBeTruthy();
  });

  it("flips the label below the box when there's no room above (e.g. near the top of frame)", () => {
    render(<CheckResultView result={FAIL_RESULT} imageUri="file:///photo.jpg" />);
    // naturalSize 1000x2000 into a 300x360 "contain" box: scale = 0.18, offsetX = 60, offsetY = 0.
    measurePhoto(300, 1000, 2000);

    // hardhat: box top = 36px, well clear of the LABEL_HEIGHT(18) threshold -> label sits above it.
    expect(screen.getByText("Hardhat 90%")).toHaveStyle({ left: 76, top: 18 });
    // no_hardhat: bbox [0,0,50,50] -> box top = 0px, no room above -> label flips below the box.
    expect(screen.getByText("No Hardhat 60%")).toHaveStyle({ left: 58, top: 11 });
  });

  it("positions the label independently of the box, so a narrow box can't clip a long label", () => {
    render(<CheckResultView result={FAIL_RESULT} imageUri="file:///photo.jpg" />);
    measurePhoto(300, 1000, 2000);

    // The label is a sibling of the box View, not nested inside it — its own
    // style carries no width at all, so it can never be stretched or
    // clipped to the (much narrower) box's width.
    const label = screen.getByText("No Hardhat 60%");
    expect(label.props.style).not.toEqual(
      expect.arrayContaining([expect.objectContaining({ width: expect.anything() })]),
    );
  });

  it("shows a fallback message instead of boxes when the photo fails to load", () => {
    render(<CheckResultView result={FAIL_RESULT} imageUri="file:///missing.jpg" />);

    fireEvent(screen.getByTestId("check-result-photo"), "error", { nativeEvent: {} });

    expect(screen.getByText("Photo is no longer available on this device.")).toBeTruthy();
    expect(screen.queryByTestId("bbox-hardhat-0")).toBeNull();
  });

  it("renders no boxes before the photo has loaded/measured", () => {
    render(<CheckResultView result={FAIL_RESULT} imageUri="file:///photo.jpg" />);

    expect(screen.queryByTestId("bbox-hardhat-0")).toBeNull();
  });
});

function makeResult(overrides: Partial<ComplianceCheckResponse> = {}): ComplianceCheckResponse {
  return {
    worker_id: "W-0042",
    job_category: "welder",
    timestamp: "2026-10-12T13:08:34Z",
    verdict: "PASS",
    confidence_avg: 0.8,
    detections: [],
    missing_equipment: [],
    violations_detected: [],
    unverifiable_equipment: [],
    explanation: null,
    model_used: "yolo11l",
    inference_time_ms: 30,
    gemini_invoked: false,
    ...overrides,
  };
}

describe("CheckResultView NEEDS_REVIEW (third verdict)", () => {
  const REVIEW = makeResult({
    verdict: "NEEDS_REVIEW",
    missing_equipment: ["safety_vest", "gloves"],
  });

  it("is amber and non-accusing — never the red FAIL treatment", () => {
    render(<CheckResultView result={REVIEW} imageUri={null} />);

    expect(screen.getByText("Needs review")).toHaveStyle({ color: "#b8860b" });
    expect(screen.getByText("?")).toHaveStyle({ color: "#b8860b" });
    expect(screen.queryByText("FAIL")).toBeNull();
    expect(screen.queryByText("✗")).toBeNull();
    // The raw enum must never leak into the UI.
    expect(screen.queryByText("NEEDS_REVIEW")).toBeNull();
  });

  it("says what couldn't be verified instead of listing it as missing equipment", () => {
    render(<CheckResultView result={REVIEW} imageUri={null} />);

    expect(
      screen.getByText("Couldn't verify: Safety Vest, Gloves — ask your supervisor to confirm."),
    ).toBeTruthy();
    expect(screen.queryByText("Missing equipment")).toBeNull();
    expect(screen.queryByText(/missing your/i)).toBeNull();
  });

  it("falls back to a generic message when nothing specific is listed", () => {
    render(
      <CheckResultView
        result={makeResult({ verdict: "NEEDS_REVIEW", missing_equipment: [] })}
        imageUri={null}
      />,
    );

    expect(
      screen.getByText("We couldn't verify this photo — ask your supervisor to confirm."),
    ).toBeTruthy();
  });

  it("renders an unrecognised future verdict as needs-review, not as a failure", () => {
    render(
      <CheckResultView
        result={makeResult({ verdict: "SOMETHING_NEW" as never, missing_equipment: ["goggles"] })}
        imageUri={null}
      />,
    );

    expect(screen.getByText("Needs review")).toBeTruthy();
    expect(screen.queryByText("FAIL")).toBeNull();
    expect(screen.queryByText("✗")).toBeNull();
  });

  it("still lists unverifiable equipment separately in amber", () => {
    render(
      <CheckResultView
        result={makeResult({
          verdict: "NEEDS_REVIEW",
          missing_equipment: ["safety_vest"],
          unverifiable_equipment: ["boots"],
        })}
        imageUri={null}
      />,
    );

    expect(screen.getByText("Verify manually")).toBeTruthy();
    expect(screen.getByText("Boots")).toHaveStyle({ color: "#b8860b" });
  });
});

describe("CheckResultView Gemini second opinion", () => {
  const VEST_SEEN = {
    model: "gemini-2.5-flash",
    elapsed_ms: 3200,
    items: [
      { item: "safety_vest", status: "worn" as const, confidence: 0.92, reason: "orange hi-vis vest on torso" },
    ],
  };

  it("shows what Gemini saw when it settles a PASS, since the vest isn't in detections", () => {
    render(
      <CheckResultView
        result={makeResult({
          verdict: "PASS",
          verdict_source: "detector+gemini",
          verification: VEST_SEEN,
          gemini_invoked: true,
        })}
        imageUri={null}
      />,
    );

    expect(screen.getByText("PASS")).toBeTruthy();
    expect(screen.getByText("What we checked")).toBeTruthy();
    expect(screen.getByText("Safety Vest — Seen")).toBeTruthy();
    expect(screen.getByText("orange hi-vis vest on torso")).toBeTruthy();
    expect(screen.getByText("Double-checked with a closer look at the photo.")).toBeTruthy();
  });

  it("shows a confirmed not-worn finding in red when Gemini decided a FAIL", () => {
    render(
      <CheckResultView
        result={makeResult({
          verdict: "FAIL",
          verdict_source: "detector+gemini",
          missing_equipment: ["safety_vest"],
          explanation: "You're missing your safety vest.",
          verification: {
            model: "gemini-2.5-flash",
            elapsed_ms: 4100,
            items: [
              { item: "safety_vest", status: "not_worn", confidence: 0.95, reason: "bare torso, t-shirt only" },
            ],
          },
          gemini_invoked: true,
        })}
        imageUri={null}
      />,
    );

    expect(screen.getByText("Missing equipment")).toBeTruthy();
    expect(screen.getByText("Safety Vest — Not seen")).toHaveStyle({ color: "#c0392b" });
    expect(screen.getByText("bare torso, t-shirt only")).toBeTruthy();
  });

  it("never presents a low-confidence not_worn as an accusation on NEEDS_REVIEW", () => {
    // The backend downgrades this finding to "unknown" (below its confidence
    // threshold) — so the app must not show it in red or quote its reason.
    render(
      <CheckResultView
        result={makeResult({
          verdict: "NEEDS_REVIEW",
          verdict_source: "detector",
          missing_equipment: ["safety_vest"],
          verification: {
            model: "gemini-2.5-flash",
            elapsed_ms: 3000,
            items: [
              { item: "safety_vest", status: "not_worn", confidence: 0.55, reason: "no vest visible" },
            ],
          },
          gemini_invoked: true,
        })}
        imageUri={null}
      />,
    );

    expect(screen.getByText("Safety Vest — Couldn't confirm")).toHaveStyle({ color: "#b8860b" });
    expect(screen.queryByText("no vest visible")).toBeNull();
    expect(screen.queryByText(/Not seen/)).toBeNull();
  });

  it("does show an uncertain finding's reason — it tells the worker how to retake", () => {
    render(
      <CheckResultView
        result={makeResult({
          verdict: "NEEDS_REVIEW",
          missing_equipment: ["gloves"],
          verification: {
            model: "gemini-2.5-flash",
            elapsed_ms: 2800,
            items: [
              { item: "gloves", status: "uncertain", confidence: 0.4, reason: "hands are out of frame" },
            ],
          },
          gemini_invoked: true,
        })}
        imageUri={null}
      />,
    );

    expect(screen.getByText("Gloves — Couldn't confirm")).toBeTruthy();
    expect(screen.getByText("hands are out of frame")).toBeTruthy();
  });

  it("renders a legacy result with no verdict_source or verification", () => {
    // History entries saved before these fields existed.
    render(<CheckResultView result={makeResult({ verdict: "PASS" })} imageUri={null} />);

    expect(screen.getByText("PASS")).toBeTruthy();
    expect(screen.queryByText("What we checked")).toBeNull();
    expect(screen.queryByText(/Double-checked/)).toBeNull();
  });
});
