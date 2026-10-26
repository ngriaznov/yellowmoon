import { Button } from "../../components";
import { CONSENT_CANCEL, CONSENT_COPY, consentBody } from "../../state/autoProviders";
import { useAppState, useStore } from "../../state/context";
import { Dialog } from "./Dialog";

/**
 * The question asked the first time Claude Code or Antigravity is chosen in Preferences (`docs/22` sections 10.2 and 12.3): what is sent
 * where, in plain words. It opens over Preferences while `consentFor` names the program, so Esc closes only it. **Use …** saves the choice
 * and the consent in one patch and closes the dialog; **Cancel**, Esc and a click outside close it and keep the previous choice. The focus
 * starts on Cancel: agreeing to send photos away takes a deliberate move.
 */
export function AutoConsent() {
  const store = useStore();
  const provider = useAppState((s) => s.consentFor);
  const os = useAppState((s) => s.os);
  if (provider === null) return null;
  const copy = CONSENT_COPY[provider];
  return (
    <Dialog
      title={copy.title}
      width={480}
      onClose={() => void store.answerConsent(false)}
      footer={
        <>
          <Button variant="secondary" data-autofocus="" onClick={() => void store.answerConsent(false)}>
            {CONSENT_CANCEL}
          </Button>
          <Button onClick={() => void store.answerConsent(true)}>{copy.agree}</Button>
        </>
      }
    >
      <p className="ym-dialog__text">{consentBody(provider, os ?? "")}</p>
    </Dialog>
  );
}
