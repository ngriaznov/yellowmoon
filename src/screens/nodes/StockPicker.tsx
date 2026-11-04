import { CheckIcon, ChevronDownIcon } from "lucide-react";
import { useState } from "react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/glass/popover";
import type { FilmCatalogue, FilmStockInfo, PhotoId } from "../../api/types";
import type { AppStore } from "../../state/store";
import { FILM_GROUPS, isoLine, stocksOf } from "./filmData";
import { useFilmTile } from "./useFilm";

interface TileProps {
  store: AppStore;
  photoId: PhotoId;
  node: number;
  stock: FilmStockInfo;
  /** What the photo is, up to the node, except the stock: the tiles are made again when it changes. */
  pictureKey: string;
  selected: boolean;
  onChoose: (stock: FilmStockInfo) => void;
}

/**
 * One stock of the picker: a small render of the open photo through it, its name and its ISO. A tile that has no picture yet shows the
 * stock's name on a plain card; the picker never waits for the pictures (`docs/24` section 4.4).
 */
function StockTile({ store, photoId, node, stock, pictureKey, selected, onChoose }: TileProps) {
  const { src, onImageError } = useFilmTile(store, photoId, node, stock.id, pictureKey, true);
  return (
    <button type="button" className="ym-look-tile ym-stock-tile" aria-pressed={selected} aria-label={`${stock.name}, ${isoLine(stock)}`} onClick={() => onChoose(stock)}>
      <span className="ym-look-tile__picture" data-ready={src ? "true" : "false"}>
        {src ? <img src={src} alt="" draggable={false} onError={onImageError} /> : <span className="ym-stock-tile__card" aria-hidden="true">{stock.name}</span>}
        {selected ? (
          <span className="ym-stock-tile__check" aria-hidden="true">
            <CheckIcon />
          </span>
        ) : null}
      </span>
      <span className="ym-stock-tile__name">{stock.name}</span>
      <span className="ym-stock-tile__iso">{isoLine(stock)}</span>
    </button>
  );
}

export interface StockPickerProps {
  store: AppStore;
  photoId: PhotoId;
  node: number;
  catalogue: FilmCatalogue;
  /** The profile id of the stock the node has. */
  value: string;
  pictureKey: string;
  onChange: (stock: string) => void;
}

/**
 * The Stock field (`docs/24` section 4.4, screen 04): a glass field that opens the tiles. Six columns, no scrolling, a small render of the
 * open photo in each of the 20 stocks, grouped Colour negative, Cine negative and Slide. The renders are asked for one at a time, in the
 * list's order, while it is open; choosing a stock closes it.
 */
export function StockPicker({ store, photoId, node, catalogue, value, pictureKey, onChange }: StockPickerProps) {
  const [open, setOpen] = useState(false);
  const current = catalogue.stocks.find((s) => s.id === value);
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button type="button" className="ym-stock-field" aria-haspopup="dialog" aria-expanded={open} aria-label={`Stock, ${current?.name ?? value}`}>
          <span className="ym-stock-field__value">
            {current?.name ?? value}
          </span>
          <ChevronDownIcon aria-hidden="true" />
        </button>
      </PopoverTrigger>
      <PopoverContent side="left" align="start" sideOffset={12} collisionPadding={16} className="ym-stock-picker" aria-label="Film stocks">
        {open
          ? FILM_GROUPS.map((group) => (
              <section key={group.kind} className="ym-stock-picker__group" aria-label={group.label}>
                <h3 className="ym-stock-picker__heading">{group.label}</h3>
                <div className="ym-stock-picker__grid">
                  {stocksOf(catalogue, group.kind).map((stock) => (
                    <StockTile
                      key={stock.id}
                      store={store}
                      photoId={photoId}
                      node={node}
                      stock={stock}
                      pictureKey={pictureKey}
                      selected={stock.id === value}
                      onChoose={(chosen) => {
                        setOpen(false);
                        if (chosen.id !== value) onChange(chosen.id);
                      }}
                    />
                  ))}
                </div>
              </section>
            ))
          : null}
      </PopoverContent>
    </Popover>
  );
}
