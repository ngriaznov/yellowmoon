/**
 * The credit lines of the film data, as the core's `film_catalogue` answers them (`docs/24` sections 4.6, 6.2 and 6.5). The name of the
 * author's project appears here and in the components that draw the credit and About, and nowhere else in the interface
 * (`tools/licences/check_film_policy.py`).
 */

/** The link on the credit line: the project's page, opened in the system browser. */
export const FILM_CREDIT_NAME = "Andrea Volpato (spektrafilm)";
/** Where the name links to. */
export const FILM_CREDIT_URL = "https://github.com/andreavolpato/spektrafilm";
/** The licence the film data is under, and its link in About. */
export const FILM_LICENCE_NAME = "CC BY-SA 4.0";
export const FILM_LICENCE_URL = "https://creativecommons.org/licenses/by-sa/4.0/";

/** The credit line. */
export const FILM_CREDIT = `Modelled on published data. Film profiles by ${FILM_CREDIT_NAME}, ${FILM_LICENCE_NAME}. Not affiliated with Kodak or Fujifilm.`;
/** The short line that goes with every stock name. */
export const FILM_DISCLAIMER = "Modelled on published data. Not affiliated with Kodak or Fujifilm.";
/** The trademark line of About. */
export const FILM_TRADEMARKS = "Kodak, Portra, Vision3, Fujifilm and Velvia are trademarks of their owners.";
