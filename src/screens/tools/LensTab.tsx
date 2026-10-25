import { memo } from "react";
import { AmountsCard, LensCard } from "./LensCard";
import type { TabProps } from "./shared";

/**
 * Lens (`docs/21` §1, `docs/11` *Lens card*): the Lens card (the seven lenses, one row each with its line) and the Amounts card
 * (Flares, Edges and Anamorphic, and the card's Reset). Off until a slider is raised: all three amounts at 0 is no lens.
 */
export const LensTab = memo(function LensTab({ store, rs }: TabProps) {
  return (
    <>
      <LensCard store={store} rs={rs} />
      <AmountsCard store={store} rs={rs} />
    </>
  );
});
