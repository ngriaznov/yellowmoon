import { MoonMark } from "../components/MoonMark";

/**
 * Placeholder for the Start screen (`design/mockups/start.html`). It carries the header's
 * folder cell (moon, name and sub-line) and says plainly that the rest is not built yet.
 * The real screen, with drag and drop and the development list, is task M3-05.
 */
export function Start() {
  return (
    <div className="ym-screen" data-screen="start">
      <header className="ym-header">
        <div className="ym-brand">
          <h1 className="ym-brand-title">
            <MoonMark />
            Yellowmoon
          </h1>
          {/* Drawn in capitals by CSS, as in the mockup; the source text stays readable. */}
          <p className="ym-brand-sub">Drop ARW files to start</p>
        </div>
        <div className="ym-header-rest" />
      </header>
      <main className="ym-screen-body">
        <div className="ym-placeholder" role="note">
          <h2 className="ym-placeholder-title">This is a placeholder.</h2>
          <p className="ym-placeholder-text">
            The Start screen, with drag and drop and the list of photos being developed, is built in milestone M3.
          </p>
        </div>
      </main>
    </div>
  );
}
