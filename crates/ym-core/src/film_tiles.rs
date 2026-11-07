//! What the Film node's panel asks the core for (`docs/24-film.md` section 4.1 and 4.4, F-09): the list of stocks and papers
//! it offers, and the **stock tiles**: a small render of the open photo through each stock.
//!
//! * **The catalogue** ([`film_catalogue`]) is `names.json` as the picker reads it: the camera films in the file's order with their
//!   kind, maker and ISO, the paper each is printed on when the settings say "Matching paper" (a cine negative's own print film is
//!   hidden, so it is the still paper that stands in, `docs/24` section 6.4), the six papers, the data version and the credit
//!   lines. The hidden rows (the two cine print films) are not in it.
//! * **A stock tile** ([`film_tile_png`], queued by [`spawn_film_tile`]) is the photo as it is developed now, with the stock of one
//!   Film node replaced by another and nothing else changed (its exposure, paper, lights, halation, grain and masks stay), through
//!   the Output as the Look tab's tiles are. It is made as a look tile is: the same size (at most 240 px on the long edge), at
//!   [`Priority::Thumbnail`], superseded when the person moves to another photo, and it leaves the photo's detail statistic alone.
//!   Nothing is written to disk; the shell keeps the pictures in memory like look tiles.

use ym_film::{StockKind, StockName};

use crate::film::{FILM_CREDIT, catalogue};
use crate::pipeline::{PhotoSource, RenderError, Renderer};
use crate::scheduler::{CancelToken, JobHandle, Priority};
use crate::settings::DevelopSettings;

/// The line that goes with every stock name (`docs/24` section 6.5).
pub const FILM_DISCLAIMER: &str =
    "Modelled on published data. Not affiliated with Kodak or Fujifilm.";

/// The trademark line of About (`docs/24` section 4.6).
pub const FILM_TRADEMARKS: &str =
    "Kodak, Portra, Ektar, Vision3, Ektachrome, Kodachrome, Fujifilm, Velvia and the other film and paper names shown are trademarks of their owners.";

/// One camera film of the picker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilmStockInfo {
    /// The profile id.
    pub id: String,
    /// The name shown to the person ("Kodak Portra 400").
    pub name: String,
    /// `colour_negative`, `cine_negative` or `slide`.
    pub kind: &'static str,
    /// The maker.
    pub group: String,
    /// The ISO number.
    pub iso: Option<u32>,
    /// The profile id of the paper it is printed on with "Matching paper"; `None` for a slide.
    pub matching_paper: Option<String>,
}

/// One still paper of the picker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilmPaperInfo {
    /// The profile id.
    pub id: String,
    /// The name shown to the person.
    pub name: String,
}

/// What the Film node's pickers list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilmCatalogue {
    /// The camera films that are offered, in the order of `names.json`.
    pub stocks: Vec<FilmStockInfo>,
    /// The still papers that are offered.
    pub papers: Vec<FilmPaperInfo>,
    /// The version of the profile data ("0.3.2").
    pub profile_version: String,
    /// The credit line (`docs/24` section 6.2).
    pub credit: String,
    /// "Modelled on published data. Not affiliated with Kodak or Fujifilm."
    pub disclaimer: String,
    /// The trademark line of About.
    pub trademarks: String,
}

fn kind_name(kind: StockKind) -> Option<&'static str> {
    match kind {
        StockKind::ColourNegative => Some("colour_negative"),
        StockKind::CineNegative => Some("cine_negative"),
        StockKind::Slide => Some("slide"),
        StockKind::Paper | StockKind::PrintFilm => None,
    }
}

fn stock_info(row: &StockName) -> Option<FilmStockInfo> {
    let cat = catalogue()?;
    Some(FilmStockInfo {
        id: row.id.clone(),
        name: row.name.clone(),
        kind: kind_name(row.kind)?,
        group: row.group.clone(),
        iso: row.iso,
        matching_paper: cat.matching_paper(&row.id),
    })
}

/// The stocks and papers the Film node offers, or `None` when the built-in `names.json` does not parse (a build error that a
/// test catches).
#[must_use]
pub fn film_catalogue() -> Option<FilmCatalogue> {
    let cat = catalogue()?;
    let stocks = cat
        .rows()
        .iter()
        .filter(|r| !r.hidden)
        .filter_map(stock_info)
        .collect();
    let papers = cat
        .rows()
        .iter()
        .filter(|r| !r.hidden && r.kind == StockKind::Paper)
        .map(|r| FilmPaperInfo {
            id: r.id.clone(),
            name: r.name.clone(),
        })
        .collect();
    Some(FilmCatalogue {
        stocks,
        papers,
        profile_version: cat.profile_version().to_owned(),
        credit: FILM_CREDIT.to_owned(),
        disclaimer: FILM_DISCLAIMER.to_owned(),
        trademarks: FILM_TRADEMARKS.to_owned(),
    })
}

/// The settings of the tile of `stock` for Film node `node`: the photo's own, with that node's stock replaced.
///
/// # Errors
///
/// [`RenderError::Color`] with a plain sentence when the photo has no such Film node or the stock is not one that is offered.
pub fn tile_settings(
    settings: &DevelopSettings,
    node: u32,
    stock: &str,
) -> Result<DevelopSettings, RenderError> {
    if let Some(cat) = catalogue()
        && let Err(sentence) = cat.check_stock(stock)
    {
        return Err(RenderError::Color(sentence));
    }
    let mut tile = settings.clone();
    let Some(film) = tile.nodes.films.get_mut(&node) else {
        return Err(RenderError::Color(format!(
            "This photo has no Film node {node}."
        )));
    };
    film.film.stock = stock.to_owned();
    Ok(tile)
}

/// The photo through `stock` in Film node `node`, as a PNG of at most 240 px on the long edge (sRGB, 8 bits), at the photo's
/// current settings and through its look.
///
/// # Errors
///
/// [`RenderError`] for a node the photo does not have, a stock that is not offered, unusable data or a cancelled job.
pub fn film_tile_png(
    renderer: &Renderer,
    src: &PhotoSource,
    node: u32,
    stock: &str,
    cancel: &CancelToken,
) -> Result<Vec<u8>, RenderError> {
    let tile = src.with_settings(tile_settings(&src.settings, node, stock)?);
    // The look tile's render leaves the photo's detail statistic alone; the photo's own look makes it the photo as it is shown.
    let look = tile.settings.nodes.output.look.clone();
    renderer.look_thumbnail_png(&tile, &look, cancel)
}

/// Queues [`film_tile_png`] at [`Priority::Thumbnail`]. Like a look tile it is a request of a panel for the photo on screen, and
/// moving to another photo cancels it, pending or running.
#[must_use]
pub fn spawn_film_tile(
    renderer: &Renderer,
    src: PhotoSource,
    node: u32,
    stock: String,
) -> JobHandle<Result<Vec<u8>, RenderError>> {
    let r = renderer.clone();
    let id = src.id;
    renderer.scheduler().submit_superseded(
        Priority::Thumbnail,
        id,
        move |token| film_tile_png(&r, &src, node, &stock, token),
        || Err(RenderError::Cancelled),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::FilmNode;
    use ym_looks::library::LookId;

    #[test]
    fn the_catalogue_lists_twenty_films_in_three_groups_and_six_papers() {
        let c = film_catalogue().expect("the built-in names.json parses");
        assert_eq!(c.stocks.len(), 20);
        let count = |k: &str| c.stocks.iter().filter(|s| s.kind == k).count();
        assert_eq!(count("colour_negative"), 11);
        assert_eq!(count("cine_negative"), 5);
        assert_eq!(count("slide"), 4);
        assert_eq!(c.papers.len(), 6);
        assert!(c.papers.iter().all(|p| !p.name.contains("Vision")));
        assert!(c.credit.contains("Andrea Volpato (spektrafilm)"));
        assert!(c.disclaimer.starts_with("Modelled on published data."));
    }

    #[test]
    fn a_cine_negative_is_matched_to_a_still_paper_and_a_slide_to_none() {
        let c = film_catalogue().expect("catalogue");
        let by = |id: &str| c.stocks.iter().find(|s| s.id == id).expect("stock");
        assert_eq!(
            by("kodak_vision3_500t").matching_paper.as_deref(),
            Some("kodak_portra_endura")
        );
        assert_eq!(
            by("fujifilm_pro_400h").matching_paper.as_deref(),
            Some("fujifilm_crystal_archive_typeii")
        );
        assert_eq!(by("fujifilm_velvia_100").matching_paper, None);
        assert_eq!(by("kodak_portra_400").iso, Some(400));
        assert_eq!(by("kodak_portra_400").kind, "colour_negative");
    }

    fn with_film(id: u32) -> DevelopSettings {
        let mut s = DevelopSettings::with_look(LookId::neutral());
        let mut node = FilmNode::default();
        node.film.exposure_ev = 0.5;
        node.grain.amount = 40.0;
        s.nodes.films.insert(id, node);
        s
    }

    #[test]
    fn a_tile_changes_the_stock_of_one_node_and_nothing_else() {
        let s = with_film(2);
        let t = tile_settings(&s, 2, "kodak_ektar_100").expect("tile");
        assert_eq!(t.nodes.films[&2].film.stock, "kodak_ektar_100");
        let mut back = t.clone();
        back.nodes.films.get_mut(&2).expect("node").film.stock =
            s.nodes.films[&2].film.stock.clone();
        assert_eq!(back, s);
        assert_eq!(
            t.nodes.films[&2].film.exposure_ev.to_bits(),
            0.5f64.to_bits()
        );
    }

    #[test]
    fn a_tile_refuses_a_node_the_photo_lacks_and_a_stock_that_is_not_offered() {
        let s = with_film(2);
        let e = tile_settings(&s, 3, "kodak_ektar_100").expect_err("no node 3");
        assert_eq!(e.to_string(), "This photo has no Film node 3.");
        let e = tile_settings(&s, 2, "kodak_2383").expect_err("a print film");
        assert!(e.to_string().starts_with("Unknown film stock"));
        let e = tile_settings(&s, 2, "kodak_portra_400_typo").expect_err("unknown");
        assert!(e.to_string().contains("ym film list"));
    }
}
