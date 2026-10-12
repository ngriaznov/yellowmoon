import { MoonMark } from "../components/MoonMark";

interface Props {
  /** Screen data attribute and heading, for example "photos". */
  screen: string;
  title: string;
  /** The milestone task that builds the real screen, for example "M3-06". */
  task: string;
  onBack: () => void;
}

/** Stands in for a screen that is routed but not built yet (Photos, Import ARRI looks). */
export function ScreenPlaceholder({ screen, title, task, onBack }: Props) {
  return (
    <div className="ym-screen" data-screen={screen}>
      <header className="ym-header">
        <div className="ym-brand">
          <h1 className="ym-brand-title">
            <MoonMark />
            {title}
          </h1>
          <p className="ym-brand-sub">Placeholder</p>
        </div>
        <div className="ym-header-rest" />
      </header>
      <main className="ym-screen-body">
        <div className="ym-placeholder" role="note">
          <h2 className="ym-placeholder-title">This screen is not built yet.</h2>
          <p className="ym-placeholder-text">It arrives with task {task}.</p>
          <div className="ym-placeholder-actions">
            <button type="button" className="ym-button" onClick={onBack}>
              Back to Start
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
