import { useEffect, useState } from "react";
import {
  Image,
  type ImageLoadEvent,
  type LayoutChangeEvent,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

import type {
  ComplianceCheckResponse,
  Detection,
  ItemVerification,
  Verdict,
} from "../api/types";

const PHOTO_HEIGHT = 360;
/** Vertical space a bbox label needs — used to decide whether it fits above the box. */
const LABEL_HEIGHT = 18;

/** "no_hardhat" -> "No Hardhat" for readable list items. */
function titleCase(key: string): string {
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/** Violation classes (no_*) are red, "person" is a neutral locator box, everything else is a compliant PPE item. */
function boxColor(detectionClass: string): string {
  if (detectionClass === "person") return "#607d8b";
  return detectionClass.startsWith("no_") ? "#c0392b" : "#1a7f37";
}

interface CheckResultViewProps {
  result: ComplianceCheckResponse;
  imageUri: string | null;
}

/**
 * Photo with detection bounding boxes overlaid — Phase 5 polish
 * (INSTRUCTIONS.md §6: "bbox overlays ... using the detections array").
 *
 * `bbox` is [x1,y1,x2,y2] in pixel coordinates of the *original* uploaded
 * image (§4), but the photo renders at whatever size `resizeMode="contain"`
 * fits into this component's box — so boxes are scaled from the image's
 * natural size (read off the `<Image>`'s own onLoad event, which reports
 * the source's true pixel dimensions regardless of how it's displayed) to
 * the container's measured width (via onLayout; height is fixed PHOTO_HEIGHT).
 */
function PhotoWithBoxes({
  imageUri,
  detections,
}: {
  imageUri: string;
  detections: Detection[];
}) {
  const [containerWidth, setContainerWidth] = useState<number | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setNaturalSize(null);
    setFailed(false);
  }, [imageUri]);

  function handleLayout(event: LayoutChangeEvent) {
    setContainerWidth(event.nativeEvent.layout.width);
  }

  function handleLoad(event: ImageLoadEvent) {
    const { width, height } = event.nativeEvent.source;
    setNaturalSize({ width, height });
  }

  if (failed) {
    return <Text style={styles.photoHint}>Photo is no longer available on this device.</Text>;
  }

  let boxes: {
    key: string;
    left: number;
    top: number;
    width: number;
    height: number;
    color: string;
    label: string;
    /** Absolute position within photoContainer — NOT nested inside the (narrow) box View,
     * so it can never inherit the box's width and get clipped/stretched to it. */
    labelLeft: number;
    labelTop: number;
  }[] = [];
  if (containerWidth && naturalSize) {
    const scale = Math.min(containerWidth / naturalSize.width, PHOTO_HEIGHT / naturalSize.height);
    const offsetX = (containerWidth - naturalSize.width * scale) / 2;
    const offsetY = (PHOTO_HEIGHT - naturalSize.height * scale) / 2;
    boxes = detections.map((d, i) => {
      const [x1, y1, x2, y2] = d.bbox;
      const top = offsetY + y1 * scale;
      const height = (y2 - y1) * scale;
      return {
        key: `${d.class}-${i}`,
        left: offsetX + x1 * scale,
        top,
        width: (x2 - x1) * scale,
        height,
        color: boxColor(d.class),
        label: `${titleCase(d.class)} ${Math.round(d.confidence * 100)}%`,
        labelLeft: offsetX + x1 * scale - 2,
        // Above the box normally; below it when there isn't room above (e.g. a hardhat near the top of frame).
        labelTop: top < LABEL_HEIGHT ? top + height + 2 : top - LABEL_HEIGHT,
      };
    });
  }

  return (
    <>
      <View testID="check-result-photo-container" style={styles.photoContainer} onLayout={handleLayout}>
        <Image
          testID="check-result-photo"
          source={{ uri: imageUri }}
          style={styles.photo}
          resizeMode="contain"
          onLoad={handleLoad}
          onError={() => setFailed(true)}
        />
        {boxes.map((box) => (
          <View
            key={box.key}
            testID={`bbox-${box.key}`}
            pointerEvents="none"
            style={[
              styles.bbox,
              { left: box.left, top: box.top, width: box.width, height: box.height, borderColor: box.color },
            ]}
          />
        ))}
        {boxes.map((box) => (
          <Text
            key={`label-${box.key}`}
            style={[
              styles.bboxLabel,
              { left: box.labelLeft, top: box.labelTop, backgroundColor: box.color },
            ]}
            numberOfLines={1}
          >
            {box.label}
          </Text>
        ))}
      </View>
      <Text style={styles.photoHint}>
        The full photo above is exactly what was sent for detection.
      </Text>
    </>
  );
}

type Tone = "pass" | "fail" | "review";

const VERDICT_PRESENTATION: Record<Verdict, { icon: string; label: string; tone: Tone }> = {
  PASS: { icon: "✓", label: "PASS", tone: "pass" },
  FAIL: { icon: "✗", label: "FAIL", tone: "fail" },
  NEEDS_REVIEW: { icon: "?", label: "Needs review", tone: "review" },
};

/**
 * Anything the backend sends that this build doesn't know about renders as
 * "needs review" — amber and non-accusing — never as a red failure. That is
 * the exact bug NEEDS_REVIEW exists to prevent (INSTRUCTIONS.md §3.1), so an
 * unrecognised future verdict must fail safe, not fall through to FAIL.
 */
function presentationFor(verdict: string) {
  return VERDICT_PRESENTATION[verdict as Verdict] ?? VERDICT_PRESENTATION.NEEDS_REVIEW;
}

/**
 * How one Gemini finding is shown to the worker.
 *
 * "not_worn" is only shown as such when the backend actually turned it into a
 * FAIL. The backend downgrades findings below its confidence threshold to
 * "unknown" without telling us the threshold, so a low-confidence "not_worn"
 * on a NEEDS_REVIEW result must read as "couldn't confirm" — and its reason
 * ("no vest visible") is withheld, since it reads as an accusation the
 * evidence doesn't support.
 */
function describeFinding(
  finding: ItemVerification,
  result: ComplianceCheckResponse,
): { label: string; tone: Tone; reason: string } {
  if (finding.status === "worn") {
    return { label: "Seen", tone: "pass", reason: finding.reason };
  }
  if (
    finding.status === "not_worn" &&
    result.verdict === "FAIL" &&
    result.missing_equipment.includes(finding.item)
  ) {
    return { label: "Not seen", tone: "fail", reason: finding.reason };
  }
  return {
    label: "Couldn't confirm",
    tone: "review",
    reason: finding.status === "uncertain" ? finding.reason : "",
  };
}

/**
 * Shared PASS / FAIL / NEEDS_REVIEW breakdown, used by both the Result screen
 * (Phase 3) and the History detail screen (Phase 4) so a past check renders
 * identically to a fresh one.
 *
 * PASS: green check + detections. FAIL: red cross + explanation +
 * missing/violations. NEEDS_REVIEW: amber, non-accusing — "couldn't verify X,
 * ask your supervisor" — because nothing was seen to be wrong.
 * unverifiable_equipment always renders separately in amber "verify
 * manually" — it is never a failure (INSTRUCTIONS.md §2, §3.1, §4).
 */
export default function CheckResultView({ result, imageUri }: CheckResultViewProps) {
  const presentation = presentationFor(result.verdict);
  const toneStyle = TONE_STYLE[presentation.tone];
  const isFail = presentation.tone === "fail";
  const isReview = presentation.tone === "review";
  const findings = result.verification?.items ?? [];

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.container}>
      {imageUri && <PhotoWithBoxes imageUri={imageUri} detections={result.detections} />}

      <View style={styles.verdictRow}>
        <Text style={[styles.verdictIcon, toneStyle]}>{presentation.icon}</Text>
        <Text style={[styles.verdictText, toneStyle]}>{presentation.label}</Text>
        {result.verdict_source === "detector+gemini" && (
          <Text style={styles.sourceHint}>Double-checked with a closer look at the photo.</Text>
        )}
      </View>

      {isFail && result.explanation && (
        <View style={styles.section}>
          <Text style={styles.explanation}>{result.explanation}</Text>
        </View>
      )}

      {isReview && (
        <View style={styles.section}>
          <Text style={styles.reviewMessage}>
            {result.missing_equipment.length > 0
              ? `Couldn't verify: ${result.missing_equipment.map(titleCase).join(", ")} — ask your supervisor to confirm.`
              : "We couldn't verify this photo — ask your supervisor to confirm."}
          </Text>
          <Text style={styles.reviewHint}>
            A clearer, full-body photo in good light often settles it.
          </Text>
        </View>
      )}

      {result.detections.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Detected</Text>
          {result.detections.map((d, i) => (
            <Text key={`${d.class}-${i}`} style={styles.listItem}>
              {titleCase(d.class)} — {Math.round(d.confidence * 100)}%
            </Text>
          ))}
        </View>
      )}

      {/* On NEEDS_REVIEW these are items nobody could confirm, already named in
          the message above — listing them again as "Missing equipment" in red
          would accuse. They only get this section when confirmed not worn. */}
      {isFail && result.missing_equipment.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Missing equipment</Text>
          {result.missing_equipment.map((item) => (
            <Text key={item} style={[styles.listItem, styles.failColor]}>
              {titleCase(item)}
            </Text>
          ))}
        </View>
      )}

      {result.violations_detected.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Violations detected</Text>
          {result.violations_detected.map((item) => (
            <Text key={item} style={[styles.listItem, styles.failColor]}>
              {titleCase(item)}
            </Text>
          ))}
        </View>
      )}

      {result.unverifiable_equipment.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Verify manually</Text>
          {result.unverifiable_equipment.map((item) => (
            <Text key={item} style={[styles.listItem, styles.amberColor]}>
              {titleCase(item)}
            </Text>
          ))}
        </View>
      )}

      {findings.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>What we checked</Text>
          {findings.map((finding) => {
            const described = describeFinding(finding, result);
            return (
              <View key={finding.item} style={styles.finding}>
                <Text style={[styles.listItem, TONE_STYLE[described.tone]]}>
                  {titleCase(finding.item)} — {described.label}
                </Text>
                {described.reason !== "" && (
                  <Text style={styles.findingReason}>{described.reason}</Text>
                )}
              </View>
            );
          })}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
  },
  container: {
    padding: 20,
    gap: 16,
  },
  photoContainer: {
    width: "100%",
    height: PHOTO_HEIGHT,
    borderRadius: 12,
    backgroundColor: "#eee",
    overflow: "hidden",
  },
  photo: {
    ...StyleSheet.absoluteFill,
  },
  bbox: {
    position: "absolute",
    borderWidth: 2,
    borderRadius: 4,
  },
  bboxLabel: {
    position: "absolute",
    // A sibling of the box View (not nested inside it), positioned with its
    // own left/top — this component previously nested the label inside the
    // (narrow) box View and relied on alignSelf to stop it stretching to
    // the box's width, but that clipped it to "Ha…" on real devices anyway.
    // Being a sibling means there's no box width for it to inherit at all.
    alignSelf: "flex-start",
    color: "#fff",
    fontSize: 10,
    fontWeight: "600",
    paddingHorizontal: 4,
    paddingVertical: 1,
    borderRadius: 3,
  },
  photoHint: {
    color: "#999",
    fontSize: 11,
    textAlign: "center",
    marginTop: -8,
  },
  verdictRow: {
    alignItems: "center",
    gap: 4,
  },
  verdictIcon: {
    fontSize: 56,
    fontWeight: "700",
  },
  verdictText: {
    fontSize: 24,
    fontWeight: "700",
  },
  passColor: {
    color: "#1a7f37",
  },
  failColor: {
    color: "#c0392b",
  },
  amberColor: {
    color: "#b8860b",
  },
  section: {
    gap: 4,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: "600",
    marginBottom: 2,
  },
  explanation: {
    fontSize: 15,
    lineHeight: 21,
  },
  sourceHint: {
    color: "#777",
    fontSize: 12,
    textAlign: "center",
    marginTop: 2,
  },
  // Darker than amberColor: this is body text, and #b8860b is too faint on the
  // screen's light-gray background to read comfortably at this size.
  reviewMessage: {
    color: "#8a6100",
    fontSize: 15,
    lineHeight: 21,
    fontWeight: "500",
  },
  reviewHint: {
    color: "#666",
    fontSize: 13,
    lineHeight: 18,
  },
  finding: {
    gap: 1,
  },
  findingReason: {
    color: "#666",
    fontSize: 12,
    lineHeight: 16,
  },
  listItem: {
    fontSize: 14,
  },
});

const TONE_STYLE: Record<Tone, object> = {
  pass: styles.passColor,
  fail: styles.failColor,
  review: styles.amberColor,
};
