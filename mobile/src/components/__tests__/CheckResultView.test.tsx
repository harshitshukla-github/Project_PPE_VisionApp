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
