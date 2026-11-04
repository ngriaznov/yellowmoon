import { Fragment } from "react";
import { LinkButton } from "../../components";
import { FILM_CREDIT, FILM_CREDIT_NAME, FILM_CREDIT_URL, FILM_LICENCE_NAME, FILM_LICENCE_URL } from "../../fixtures/filmCredit";
import type { AppStore } from "../../state/store";

export interface FilmCreditProps {
  store: AppStore;
  /** The credit line the core gave (`film_catalogue`); the line the page knows until it has. */
  credit?: string;
  /** About also links the licence ("CC BY-SA 4.0"); the Film tab links the project's name only. */
  licenceLink?: boolean;
  className?: string;
}

/**
 * The credit line of the film data (`docs/24` sections 4.6 and 6.2): "Modelled on published data. Film profiles by Andrea Volpato
 * (…), CC BY-SA 4.0. Not affiliated with Kodak or Fujifilm." The author's name is a link that opens the project's page in the system
 * browser, like ARRI's page does; in About the licence is a link too. The Film tab ends with it and About carries it.
 */
export function FilmCredit({ store, credit = FILM_CREDIT, licenceLink = false, className }: FilmCreditProps) {
  const open = (url: string): void => void store.api.openUrl(url).catch((error) => store.reportError(error));
  const links: ReadonlyArray<[string, string]> = licenceLink
    ? [
        [FILM_CREDIT_NAME, FILM_CREDIT_URL],
        [FILM_LICENCE_NAME, FILM_LICENCE_URL],
      ]
    : [[FILM_CREDIT_NAME, FILM_CREDIT_URL]];
  // The line is cut at each name that is a link, in the order they stand in it.
  const parts: Array<string | [string, string]> = [];
  let rest = credit;
  while (rest.length > 0) {
    const found = links.map(([name, url]) => ({ name, url, at: rest.indexOf(name) })).filter((f) => f.at >= 0).sort((a, b) => a.at - b.at)[0];
    if (!found) {
      parts.push(rest);
      break;
    }
    if (found.at > 0) parts.push(rest.slice(0, found.at));
    parts.push([found.name, found.url]);
    rest = rest.slice(found.at + found.name.length);
  }
  return (
    <p className={className ?? "ym-film-credit"}>
      {parts.map((part, i) =>
        typeof part === "string" ? (
          <Fragment key={i}>{part}</Fragment>
        ) : (
          <LinkButton key={i} size="sm" weight="bold" className="ym-film-credit__link" onClick={() => open(part[1])}>
            {part[0]}
          </LinkButton>
        ),
      )}
    </p>
  );
}
