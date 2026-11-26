import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { Page } from "../api/types";
import { usePages } from "./useResource";

/** A list of 3 pages of 2; the cursor is the page number as text. */
function fakeList() {
  return vi.fn(async (cursor: string | null): Promise<Page<number>> => {
    const n = cursor === null ? 0 : Number(cursor);
    return { items: [n * 2 + 1, n * 2 + 2], next_cursor: n < 2 ? String(n + 1) : null, total: 6 };
  });
}

describe("usePages: one load per change", () => {
  it("loads the first page once when it opens", async () => {
    const load = fakeList();
    const filters = { site: "" };

    const { result } = renderHook(() => usePages((cursor) => load(cursor), 2, [filters]));

    await waitFor(() => expect(result.current.page?.items).toEqual([1, 2]));
    expect(load).toHaveBeenCalledTimes(1);
  });

  it("loads once, from the first page, when the filters change", async () => {
    const load = fakeList();
    const { result, rerender } = renderHook(({ filters }) => usePages((cursor) => load(cursor), 2, [filters]), {
      initialProps: { filters: { site: "" } },
    });
    await waitFor(() => expect(result.current.page).not.toBeNull());
    load.mockClear();

    rerender({ filters: { site: "pune_tower_b" } });

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(load).toHaveBeenCalledTimes(1);
    expect(load).toHaveBeenCalledWith(null);
  });

  it("from page 2, a filter change loads the first page once, never page 2 with the new filters", async () => {
    const load = fakeList();
    const { result, rerender } = renderHook(({ filters }) => usePages((cursor) => load(cursor), 2, [filters]), {
      initialProps: { filters: { site: "" } },
    });
    await waitFor(() => expect(result.current.hasNext).toBe(true));
    act(() => {
      result.current.next();
    });
    await waitFor(() => expect(result.current.range?.first).toBe(3));
    load.mockClear();

    rerender({ filters: { site: "pune_tower_b" } });

    await waitFor(() => expect(result.current.range?.first).toBe(1));
    expect(load.mock.calls).toEqual([[null]]);
    expect(result.current.hasPrevious).toBe(false);
  });

  it("still pages forward and back, one load each", async () => {
    const load = fakeList();
    const { result } = renderHook(() => usePages((cursor) => load(cursor), 2, ["same"]));
    await waitFor(() => expect(result.current.hasNext).toBe(true));
    load.mockClear();

    act(() => {
      result.current.next();
    });
    await waitFor(() => expect(result.current.range).toEqual({ first: 3, last: 4, total: 6 }));
    act(() => {
      result.current.previous();
    });
    await waitFor(() => expect(result.current.range).toEqual({ first: 1, last: 2, total: 6 }));

    expect(load.mock.calls).toEqual([["1"], [null]]);
  });
});
