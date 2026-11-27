import { useCallback, useEffect, useState, type DependencyList } from "react";

import { problemFrom, type Problem } from "../api/problems";
import type { Page } from "../api/types";

interface ResourceState<T> {
  data: T | null;
  problem: Problem | null;
  loading: boolean;
}

/**
 * Loads something when the page opens (and again when `deps` change or
 * `reload()` is called). Keeps the last good data on screen while a reload
 * is running or has failed, so a failed refresh never blanks a page.
 */
export function useResource<T>(load: () => Promise<T>, deps: DependencyList) {
  const [state, setState] = useState<ResourceState<T>>({ data: null, problem: null, loading: true });
  const [round, setRound] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, problem: null }));
    load().then(
      (data) => {
        if (!cancelled) setState({ data, problem: null, loading: false });
      },
      (err: unknown) => {
        if (!cancelled) {
          setState((s) => ({ data: s.data, problem: problemFrom(err, "Couldn't load this page."), loading: false }));
        }
      },
    );
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, round]);

  const reload = useCallback(() => setRound((r) => r + 1), []);
  const setData = useCallback(
    (update: T | ((previous: T | null) => T)) =>
      setState((s) => ({ ...s, data: typeof update === "function" ? (update as (p: T | null) => T)(s.data) : update })),
    [],
  );
  return { ...state, reload, setData };
}

/**
 * A paged list (docs/admin-api.md §2): forward by opaque cursor, and back by
 * remembering the cursors already used. Changing `deps` (the filters) starts
 * again from the first page.
 */
export function usePages<T>(
  load: (cursor: string | null) => Promise<Page<T>>,
  pageSize: number,
  deps: DependencyList,
) {
  const [cursors, setCursors] = useState<Array<string | null>>([null]);
  const [filters, setFilters] = useState<DependencyList>(deps);

  // New filters: back to the first page, decided while rendering rather than in an effect
  // afterwards. Resetting in an effect loaded twice: once with the old page's cursor (page 2's,
  // with the new filters), then again from page 1 when the reset landed.
  const filtersChanged = !sameDeps(filters, deps);
  if (filtersChanged) {
    setFilters(deps);
    setCursors([null]);
  }
  const current = filtersChanged ? [null] : cursors;
  const index = current.length - 1;
  const resource = useResource(() => load(current[index]), [...deps, current.length, current[index]]);

  const page = resource.data;
  const first = page && page.items.length ? index * pageSize + 1 : 0;
  return {
    ...resource,
    page,
    range: page ? { first, last: first ? first + page.items.length - 1 : 0, total: page.total } : null,
    hasPrevious: index > 0,
    hasNext: !!page?.next_cursor,
    next: () => page?.next_cursor && setCursors((c) => [...c, page.next_cursor]),
    previous: () => setCursors((c) => (c.length > 1 ? c.slice(0, -1) : c)),
  };
}

/** Whether two dependency lists hold the same values, compared as React compares effect deps. */
function sameDeps(a: DependencyList, b: DependencyList): boolean {
  return a.length === b.length && a.every((value, i) => Object.is(value, b[i]));
}
