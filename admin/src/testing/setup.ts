import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Vitest doesn't unmount rendered components between tests by itself.
afterEach(() => cleanup());
