import { Component, type ErrorInfo, type ReactNode } from "react";

/**
 * Catches an error thrown while drawing a page, so one broken page (an answer
 * it didn't expect, say) shows a message instead of blanking the whole
 * dashboard. Inside the frame, the navigation still works; give it a `key`
 * per page so that opening the page again tries it afresh.
 *
 * It only sees errors thrown while rendering. A failed request is already
 * turned into a message by the page itself.
 */
export class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }

  componentDidCatch(error: unknown, info: ErrorInfo): void {
    // For whoever opens the browser console; React reports it there too, without the page's name.
    console.error("The dashboard couldn't draw this page.", error, info.componentStack);
  }

  render(): ReactNode {
    if (!this.state.failed) return this.props.children;
    return (
      <section className="page" role="alert">
        <h1>Something went wrong on this page</h1>
        <p>
          The dashboard couldn&apos;t show it. Anything you were typing on it wasn&apos;t saved. Reload the page to
          start it again, or go to another page.
        </p>
        <button type="button" onClick={() => window.location.reload()}>
          Reload the page
        </button>
      </section>
    );
  }
}
