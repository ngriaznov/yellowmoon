/**
 * Compile-time checks that `types.ts` (what the screens are written against) and `bindings.ts` (what `tauri-specta` makes of the Rust
 * commands and types) describe the same data. There is no run-time code here: `tsc -b` (part of `npm run build`) fails when a type is
 * added, removed or changed on one side only.
 *
 * What is checked:
 *  - every payload that crosses the bridge is assignable the way it flows: answers and events, generated to hand-written
 *    (`FromCore`); arguments, hand-written to generated (`ToCore`); what goes both ways, in both (`Both`);
 *  - every event the webview listens to is one of the generated `events`, with the payload the generated one has, and the shell
 *    sends no event the webview does not know (`Events`);
 *  - every method of `Api` is a generated command that takes the arguments and answers what the command does (`Commands`).
 *
 * Deliberate differences, each stated where it is excluded: `SettingsPatch` (the core accepts more than `Partial<DevelopSettings>`, so
 * the narrower one is what the screens send), `UpdateState` (the shell never sends it; the updater runs in the webview), the unit
 * commands (`null` on the wire, `void` in `Api`), `looksImport` (an optional `input`), `viewportInfo` (the mock has no viewport) and
 * `copySettings` and `pasteSettings` (they answer patches, which the generated type allows `null` in every field of; `tauriApi` narrows
 * them, see `narrowed` in `api.ts`).
 */
import type { Api } from "../api";
import type { commands, events } from "./bindings";
import type * as B from "./bindings";
import type * as T from "./types";

/** `true` when `A` can be used where `Target` is expected. */
type Assignable<A, Target> = [A] extends [Target] ? true : false;
/** `true` when each can be used for the other. */
type Same<A, Other> = Assignable<A, Other> extends true ? Assignable<Other, A> : false;
/** A compile error at the alias unless every value of the record is `true`. */
type AllTrue<Checks extends Record<string, true>> = Checks;

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

/** Answers and events: what the core sends has to fit the type the screens read. */
export type FromCore = AllTrue<{
  sessionState: Assignable<B.SessionState, T.SessionState>;
  sessionChange: Assignable<B.SessionChange, T.SessionChange>;
  photoSummary: Assignable<B.PhotoSummary, T.PhotoSummary>;
  photoStatus: Assignable<B.PhotoStatus, T.PhotoStatus>;
  cameraInfo: Assignable<B.CameraInfo, T.CameraInfo>;
  resolvedSettings: Assignable<B.ResolvedSettings, T.ResolvedSettings>;
  developSettings: Assignable<B.DevelopSettings, T.DevelopSettings>;
  autoEstimates: Assignable<B.AutoEstimates, T.AutoEstimates>;
  appleAuto: Assignable<B.AppleAuto, T.AppleAuto>;
  appleAutoStatus: Assignable<B.AppleAutoStatus, T.AppleAutoStatus>;
  look: Assignable<B.Look, T.Look>;
  looksImportResult: Assignable<B.LooksImportResult, T.LooksImportResult>;
  sourceRecognition: Assignable<B.SourceRecognition, T.SourceRecognition>;
  displayLut: Assignable<B.DisplayLut, T.DisplayLut>;
  watchStatus: Assignable<B.WatchStatus, T.WatchStatus>;
  pixelSample: Assignable<B.PixelSample, T.PixelSample>;
  histogramData: Assignable<B.HistogramData, T.HistogramData>;
  chromaScatter: Assignable<B.ChromaScatter, T.ChromaScatter>;
  chromaSample: Assignable<B.ChromaSample, T.ChromaSample>;
  exportEstimate: Assignable<B.ExportEstimate, T.ExportEstimate>;
  cliLinkResult: Assignable<B.CliLinkResult, T.CliLinkResult>;
  cliLinkStatus: Assignable<B.CliLinkStatus, T.CliLinkStatus>;
  appInfo: Assignable<B.AppInfo, T.AppInfo>;
  viewportInfo: Assignable<B.ViewportInfo, T.ViewportInfo>;
  appError: Assignable<B.AppError, T.AppError>;
  menuAction: Assignable<B.MenuAction, T.MenuAction>;
  openSummary: Assignable<B.OpenSummary, T.OpenSummary>;
}>;

/** Arguments: what the screens send has to fit the type the core takes. */
export type ToCore = AllTrue<{
  // The Rust patch is wider (nested objects may be partial, `null` clears): the screens send the narrower one.
  settingsPatch: Assignable<T.SettingsPatch, B.SettingsPatch>;
  preferencesPatch: Assignable<T.PreferencesPatch, B.PreferencesPatch>;
  lookInput: Assignable<T.LookInput, NonNullable<Parameters<typeof commands.looksImport>[1]>>;
  openKind: Assignable<T.OpenKind, B.OpenKind>;
  applyScope: Assignable<T.ApplyScope, B.ApplyScope>;
  menuState: Assignable<T.MenuState, B.MenuState>;
}>;

/** Types that go both ways, or that both sides build: they have to be the same type. */
export type Both = AllTrue<{
  exportOptions: Same<T.ExportOptions, B.ExportOptions>;
  exportJob: Same<T.ExportJob, B.ExportJob>;
  preferences: Same<T.Preferences, B.Preferences>;
  lastUsedFolders: Same<T.LastUsedFolders, B.LastUsedFolders>;
  displayRendering: Same<T.DisplayRendering, B.DisplayRendering>;
  crop: Same<T.Crop, B.Crop>;
  curves: Same<T.Curves, B.Curves>;
  colorBalance: Same<T.ColorBalance, B.ColorBalance>;
  colorBalancePoints: Same<T.ColorBalancePoints, B.ColorBalancePoints>;
  balanceSetting: Same<T.BalanceSetting, B.BalanceSetting>;
  colorWarp: Same<T.ColorWarp, B.ColorWarp>;
  colorWarpPin: Same<T.ColorWarpPin, B.ColorWarpPin>;
  viewportRect: Same<T.ViewportRect, B.ViewportRect>;
  viewportInput: Same<T.ViewportInput, B.ViewportInput>;
  viewportMode: Same<T.ViewportMode, B.ViewportMode>;
  viewportFrame: Same<T.ViewportFrame, B.ViewportFrame>;
  viewportBackendKind: Same<T.ViewportBackendKind, B.ViewportBackendKind>;
  errorScope: Same<T.ErrorScope, B.ErrorScope>;
  lightSource: Same<T.LightSource, B.LightSource>;
  percentSetting: Same<T.PercentSetting, B.PercentSetting>;
  autoSource: Same<T.AutoSource, B.AutoSource>;
  appleAutoReason: Same<T.AppleAutoReason, B.AppleAutoReason>;
  appleAutoFailure: Same<T.AppleAutoFailure, B.AppleAutoFailure>;
  appleAutoMode: Same<T.AppleAutoMode, B.AppleAutoMode>;
  appleAutoInfo: Same<T.AppleAutoInfo, B.AppleAutoInfo>;
  appleAutoSteps: Same<T.AppleAutoSteps, B.AppleAutoSteps>;
  appleAutoAnswer: Same<T.AppleAutoAnswer, B.AppleAutoAnswer>;
  appleAutoInstructions: Same<T.AppleAutoInstructions, B.AppleAutoInstructions>;
  appleShadowColor: Same<T.AppleShadowColor, B.AppleShadowColor>;
  appleHighlightColor: Same<T.AppleHighlightColor, B.AppleHighlightColor>;
  appleAutoStatus: Same<T.AppleAutoStatus, B.AppleAutoStatus>;
  lookInput: Same<T.LookInput, B.LookInput>;
  lookGroup: Same<T.LookGroup, B.LookGroup>;
  lookLibrary: Same<T.LookLibrary, B.LookLibrary>;
  lookFormat: Same<T.LookFormat, B.LookFormat>;
  packageId: Same<T.PackageId, B.PackageId>;
  colorSpace: Same<T.ColorSpace, B.ColorSpace>;
  exportFormat: Same<T.ExportFormat, B.ExportFormat>;
  exportSize: Same<T.ExportSize, B.ExportSize>;
  ifExists: Same<T.IfExists, B.IfExists>;
  tiffCompression: Same<T.TiffCompression, B.TiffCompression>;
  photoStage: Same<T.PhotoStage, B.PhotoStage>;
  matchKind: Same<T.MatchKind, B.MatchKind>;
  recognition: Same<T.Recognition, B.Recognition>;
  menuAction: Same<T.MenuAction, B.MenuAction>;
  menuState: Same<T.MenuState, B.MenuState>;
}>;

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/** "photo_progress" becomes "photoProgress", the key the generated `events` object uses. */
type Camel<Name extends string> = Name extends `${infer Head}_${infer Tail}` ? `${Head}${Capitalize<Camel<Tail>>}` : Name;

/** The payload a generated event delivers to its listener. */
type PayloadOf<Event> = Event extends { listen: (callback: (event: { payload: infer Payload }) => unknown) => unknown } ? Payload : never;

/** The events the webview sends itself rather than receiving from the shell. */
type OwnEvents = "update_state";

/**
 * Every event of `ApiEvents` (but `update_state`, which the shell never sends) is a generated event whose payload fits what the screens
 * read, and every generated event is an event of `ApiEvents`.
 */
export type Events = AllTrue<
  {
    [Name in Exclude<T.ApiEventName, OwnEvents>]: Camel<Name> extends keyof typeof events
      ? Assignable<PayloadOf<(typeof events)[Camel<Name>]>, T.ApiEvents[Name]>
      : false;
  } & {
    [Key in keyof typeof events]: Key extends Camel<T.ApiEventName> ? true : false;
  }
>;

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/** What a generated command resolves to; `null` is the answer of a unit command. */
type Answer<Fn> = Fn extends (...args: never[]) => Promise<infer Result> ? Result : never;
type Args<Fn> = Fn extends (...args: infer Parameters) => unknown ? Parameters : never;

/** Commands whose `Api` method differs from the generated one on purpose. */
type Adapted = "looksImport" | "viewportInfo" | "copySettings" | "pasteSettings";

/**
 * Every generated command has an `Api` method of the same name that passes its arguments on (the method's arguments fit the
 * command's) and whose answer is what the command answers (a unit command answers `void`), and every `Api` method but `on` is a
 * generated command.
 */
export type Commands = AllTrue<
  {
    [Name in keyof typeof commands]: Name extends Adapted
      ? true
      : Name extends keyof Api
        ? Assignable<Args<Api[Name]>, Args<(typeof commands)[Name]>> extends true
          ? Answer<(typeof commands)[Name]> extends null
            ? true
            : Assignable<Answer<(typeof commands)[Name]>, Answer<Api[Name]>>
          : false
        : false;
  } & {
    [Name in Exclude<keyof Api, "on">]: Name extends keyof typeof commands ? true : false;
  }
>;
