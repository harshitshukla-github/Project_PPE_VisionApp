import { CameraType, CameraView, useCameraPermissions } from "expo-camera";
import { useRouter } from "expo-router";
import * as ImagePicker from "expo-image-picker";
import { useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";

import { ApiError, postComplianceCheck } from "../../src/api/client";
import { problemFrom, type Problem } from "../../src/api/problems";
import { useSession } from "../../src/context/SessionContext";

type CaptureStatus =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "error"; problem: Problem };

/**
 * Capture screen — Phase 3 (demo milestone).
 *
 * Camera preview (front-facing default, per INSTRUCTIONS.md §2) with a
 * full-body framing hint, plus a gallery fallback for emulator/desktop
 * testing. Either source POSTs to /compliance/check and navigates to
 * Result on success.
 */
export default function CaptureScreen() {
  const router = useRouter();
  const { jobCategory, jobNotice, refreshJobs, recordCheck, expireSession, requirePinChange } =
    useSession();
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<CameraType>("front");
  const [status, setStatus] = useState<CaptureStatus>({ kind: "idle" });
  const cameraRef = useRef<CameraView>(null);

  const identityReady = !!jobCategory;

  async function submitPhoto(imageUri: string) {
    setStatus({ kind: "checking" });
    try {
      // An admin may have deactivated this job since the list was loaded
      // (docs/admin-api.md §2). Look again before sending the photo; if the
      // job has gone, refreshJobs clears the selection and Capture asks the
      // worker to pick again. If the server can't be asked (null), go on and
      // let the check itself report the problem.
      const jobs = await refreshJobs();
      if (jobs && !jobs[jobCategory as string]) {
        setStatus({ kind: "idle" });
        return;
      }
      const result = await postComplianceCheck({
        imageUri,
        jobCategory: jobCategory as string,
      });
      recordCheck(result, imageUri);
      setStatus({ kind: "idle" });
      router.push("/result");
    } catch (err) {
      // The session itself is the problem: hand over to the root layout, which
      // opens Login (expired) or Change PIN (still on the issued PIN).
      if (err instanceof ApiError && err.code === "not_authenticated") {
        setStatus({ kind: "idle" });
        expireSession();
        return;
      }
      if (err instanceof ApiError && err.code === "pin_change_required") {
        setStatus({ kind: "idle" });
        requirePinChange();
        return;
      }
      // The job went between our look and the check. Look again: if it's gone,
      // the selection is cleared with a note; if the list still has it (another
      // server process hasn't caught up yet), ask the worker to try again.
      if (err instanceof ApiError && err.code === "unknown_job") {
        const jobs = await refreshJobs();
        const gone = jobs !== null && !jobs[jobCategory as string];
        setStatus(
          gone
            ? { kind: "idle" }
            : {
                kind: "error",
                problem: {
                  message: "That job isn't available right now. Try again shortly, or pick your job again on Home.",
                  tone: "notice",
                },
              },
        );
        return;
      }
      setStatus({
        kind: "error",
        problem: problemFrom(err, "Unexpected error checking compliance."),
      });
    }
  }

  async function handleShutterPress() {
    if (!cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.7 });
      if (photo?.uri) {
        await submitPhoto(photo.uri);
      }
    } catch {
      setStatus({
        kind: "error",
        problem: { message: "Couldn't take that photo — try again.", tone: "error" },
      });
    }
  }

  async function handlePickFromGallery() {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert(
        "Photo access needed",
        "Allow photo library access in your device settings to pick a test image.",
      );
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      quality: 0.7,
    });
    if (!result.canceled && result.assets[0]?.uri) {
      await submitPhoto(result.assets[0].uri);
    }
  }

  function toggleFacing() {
    setFacing((current) => (current === "front" ? "back" : "front"));
  }

  if (!identityReady) {
    return (
      <View style={styles.center}>
        <Text style={styles.title}>Capture</Text>
        <Text style={[styles.subtitle, jobNotice && styles.noticeText]}>
          {jobNotice ? `${jobNotice.message} Go to Home.` : "Pick a job category on Home first."}
        </Text>
      </View>
    );
  }

  if (!permission) {
    return (
      <View style={styles.center}>
        <ActivityIndicator />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.title}>Capture</Text>
        <Text style={styles.subtitle}>
          Camera access is needed to take a compliance photo.
        </Text>
        <Pressable style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant camera access</Text>
        </Pressable>
        <Pressable onPress={handlePickFromGallery}>
          <Text style={styles.linkText}>Or pick a photo from your gallery</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing={facing} ref={cameraRef} />

      {/* SDK 57's CameraView no longer renders children (it silently drops
          them) — all overlay UI must be a sibling, absolutely positioned
          on top. See node_modules/expo-camera/src/CameraView.tsx. */}
      <View style={styles.overlayRoot} pointerEvents="box-none">
        <View style={styles.overlay} pointerEvents="none">
          <View style={styles.framingGuide} />
          <Text style={styles.framingHint}>Fit your full body in frame</Text>
        </View>

        {status.kind === "checking" && (
          <View style={styles.checkingOverlay}>
            <ActivityIndicator color="#fff" size="large" />
            <Text style={styles.checkingText}>Checking compliance…</Text>
          </View>
        )}

        {status.kind === "error" && (
          <View
            style={[
              styles.errorBanner,
              status.problem.tone === "notice" && styles.noticeBanner,
            ]}
          >
            <Text style={styles.errorBannerText}>{status.problem.message}</Text>
          </View>
        )}

        <View style={styles.controls}>
          <Pressable style={styles.secondaryButton} onPress={handlePickFromGallery}>
            <Text style={styles.secondaryButtonText}>Gallery</Text>
          </Pressable>

          <Pressable
            testID="shutter-button"
            style={styles.shutter}
            onPress={handleShutterPress}
            disabled={status.kind === "checking"}
          />

          <Pressable style={styles.secondaryButton} onPress={toggleFacing}>
            <Text style={styles.secondaryButtonText}>Flip</Text>
          </Pressable>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 12,
    padding: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: "600",
  },
  subtitle: {
    color: "#666",
    textAlign: "center",
  },
  noticeText: {
    color: "#8a5a0b",
    fontWeight: "500",
  },
  camera: {
    ...StyleSheet.absoluteFill,
  },
  overlayRoot: {
    ...StyleSheet.absoluteFill,
  },
  overlay: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  framingGuide: {
    width: "60%",
    height: "70%",
    borderWidth: 2,
    borderColor: "rgba(255,255,255,0.8)",
    borderStyle: "dashed",
    borderRadius: 16,
  },
  framingHint: {
    color: "#fff",
    marginTop: 12,
    fontSize: 14,
    fontWeight: "500",
    textShadowColor: "rgba(0,0,0,0.6)",
    textShadowRadius: 4,
  },
  checkingOverlay: {
    ...StyleSheet.absoluteFill,
    backgroundColor: "rgba(0,0,0,0.55)",
    alignItems: "center",
    justifyContent: "center",
    gap: 12,
  },
  checkingText: {
    color: "#fff",
    fontSize: 15,
    fontWeight: "500",
  },
  errorBanner: {
    position: "absolute",
    top: 16,
    left: 16,
    right: 16,
    backgroundColor: "rgba(192,57,43,0.92)",
    borderRadius: 8,
    padding: 12,
  },
  noticeBanner: {
    backgroundColor: "rgba(185,119,14,0.94)",
  },
  errorBannerText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "500",
  },
  controls: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 32,
    paddingBottom: 32,
  },
  shutter: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: "#fff",
    borderWidth: 4,
    borderColor: "rgba(255,255,255,0.5)",
  },
  secondaryButton: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: "rgba(0,0,0,0.45)",
  },
  secondaryButtonText: {
    color: "#fff",
    fontWeight: "600",
  },
  button: {
    backgroundColor: "#1d6fd6",
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 20,
    alignItems: "center",
  },
  buttonText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 15,
  },
  linkText: {
    color: "#1d6fd6",
    fontWeight: "500",
  },
});
