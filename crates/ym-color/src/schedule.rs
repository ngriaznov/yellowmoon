//! The scheduler's area inference (`docs/23` section 5.12, `docs/17` section 18).
//!
//! Every step of a render declares a [`Footprint`]: per pixel, or spatial with a reach in output pixels, and what it
//! reads ([`Reads`]). A [`Schedule`] is built from those declarations alone (it never sees a node or mask kind) and
//! answers one question for any graph: *given the area of the output that is wanted, which area does every spatial
//! step make, and which area of the camera picture is read?* The CPU path and the device path both ask it, so they
//! agree on areas by construction, and a tile equals the same pixels of the whole frame.
//!
//! The rules (all in absolute pixels of the photo's extent, `frame`):
//!
//! * The Output is per pixel: it reads every picture and plane over `need`.
//! * A spatial step makes the union of what its consumers read, cut to `frame`, and reads its inputs over that grown by
//!   its reach, cut to `frame`.
//! * The mask planes are one shared value (a plane is made once however many rows read it): their reach is the largest
//!   reach declared, and they are made over the union of their consumers' reads.
//! * Pictures (a Film node's) are made in plan order; a step that reads pictures reads every earlier one over its own
//!   read area.

use crate::mask_raster::Area;

/// What a spatial step makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Produces {
    /// A picture at the node's output (a Film node's), made in plan order.
    Picture,
    /// A mask plane. All planes are one shared value, made once over the union of what reads them.
    Plane,
}

/// How a step reads its neighbours.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Footprint {
    /// Each output pixel depends on the same pixel of its inputs.
    PerPixel,
    /// Each output pixel depends on the inputs within `reach_px` output pixels. The result must not depend on the
    /// window it is made in (tile invariance).
    Spatial {
        /// The reach in output pixels at this render's scale.
        reach_px: usize,
        /// The value the step makes.
        produces: Produces,
    },
}

/// What a step reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Reads {
    /// The camera picture (the Input's pixels, sampled).
    pub source: bool,
    /// The mask planes.
    pub planes: bool,
    /// The pictures earlier steps made.
    pub pictures: bool,
}

/// A step's declaration: its footprint and what it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Declared {
    /// How far it reads.
    pub footprint: Footprint,
    /// What it reads.
    pub reads: Reads,
}

/// What a plan step is. The scheduler's one list of step kinds besides mask rows (`crate::mask::OpKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepKind {
    /// A corrector (or Main): per pixel.
    Corrector,
    /// A Film node that does something: spatial.
    Film,
}

impl StepKind {
    /// Every step kind. A `match`-checked list: [`StepKind::declared`] does not compile when a kind is missing, and the
    /// unit test below checks this list against it.
    pub const ALL: [StepKind; 2] = [StepKind::Corrector, StepKind::Film];

    /// The kind's footprint. **Exhaustive on purpose** (no wildcard arm): a new step kind does not compile until it says
    /// whether it is per pixel or spatial, how far it reads (`reach_px`, from its own data) and what it reads.
    #[must_use]
    pub fn declared(self, reach_px: usize) -> Declared {
        match self {
            Self::Corrector => Declared {
                footprint: Footprint::PerPixel,
                reads: Reads {
                    pictures: true,
                    planes: true,
                    ..Reads::default()
                },
            },
            Self::Film => Declared {
                footprint: Footprint::Spatial {
                    reach_px,
                    produces: Produces::Picture,
                },
                reads: Reads {
                    source: true,
                    planes: true,
                    pictures: true,
                },
            },
        }
    }
}

/// The areas [`Schedule::infer`] found.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Areas {
    /// The area of the camera picture to sample, cut to the frame.
    pub source: Area,
    /// The area the mask planes are made over (empty when no step makes planes).
    pub planes: Area,
    /// For each picture step in plan order: the area it makes.
    pub made: Vec<Area>,
    /// For each picture step in plan order: the area of its inputs it reads.
    pub read: Vec<Area>,
}

/// The declared steps of a render, in the plan's order, ready to infer areas.
#[derive(Debug, Clone, Default)]
pub struct Schedule {
    plane_reach: Option<usize>,
    plane_reads_source: bool,
    pictures: Vec<(usize, Reads)>,
    output: Reads,
}

fn union(a: Option<Area>, b: Area) -> Option<Area> {
    if b.is_empty() {
        return a;
    }
    Some(match a {
        None => b,
        Some(a) => {
            let (x, y) = (a.x.min(b.x), a.y.min(b.y));
            let (x1, y1) = (a.x1().max(b.x1()), a.y1().max(b.y1()));
            Area {
                x,
                y,
                w: (x1 - x) as usize,
                h: (y1 - y) as usize,
            }
        }
    })
}

/// The smallest area holding both (an empty area is ignored).
#[must_use]
pub fn union_area(a: Area, b: Area) -> Area {
    union((!a.is_empty()).then_some(a), b)
        .or((!a.is_empty()).then_some(a))
        .unwrap_or_default()
}

impl Schedule {
    /// The schedule of steps declared in plan order. A per-pixel step reads over the output's area, so its reads fold into
    /// the Output's.
    #[must_use]
    pub fn new(steps: impl IntoIterator<Item = Declared>) -> Self {
        let mut s = Self {
            output: Reads {
                source: true,
                planes: false,
                pictures: false,
            },
            ..Self::default()
        };
        for d in steps {
            match d.footprint {
                Footprint::PerPixel => {
                    s.output.source |= d.reads.source;
                    s.output.planes |= d.reads.planes;
                    s.output.pictures |= d.reads.pictures;
                }
                Footprint::Spatial {
                    reach_px,
                    produces: Produces::Plane,
                } => {
                    s.plane_reach = Some(s.plane_reach.unwrap_or(0).max(reach_px));
                    s.plane_reads_source |= d.reads.source;
                    s.output.planes = true;
                }
                Footprint::Spatial {
                    reach_px,
                    produces: Produces::Picture,
                } => {
                    s.pictures.push((reach_px, d.reads));
                    s.output.pictures = true;
                }
            }
        }
        s
    }

    /// Whether any step is spatial.
    #[must_use]
    pub fn is_spatial(&self) -> bool {
        self.plane_reach.is_some() || !self.pictures.is_empty()
    }

    /// The reach of the whole schedule: how far the camera picture is read beyond the output. For a schedule whose steps
    /// all read the camera picture it is the plane reach plus every picture step's reach (a step reads the one before it
    /// over its own reach).
    #[must_use]
    pub fn total_reach(&self) -> usize {
        self.plane_reach.unwrap_or(0) + self.pictures.iter().map(|p| p.0).sum::<usize>()
    }

    /// The areas for an output area `need` (absolute pixels, not cut) of a photo extent `frame`.
    #[must_use]
    pub fn infer(&self, need: Area, frame: &Area) -> Areas {
        let n = self.pictures.len();
        let mut pic_demand: Vec<Option<Area>> = vec![None; n];
        let mut plane_demand = None;
        let mut source = None;
        if self.output.pictures {
            for d in &mut pic_demand {
                *d = union(*d, need);
            }
        }
        if self.output.planes {
            plane_demand = union(plane_demand, need);
        }
        if self.output.source {
            source = union(source, need);
        }
        let mut made = vec![Area::default(); n];
        let mut read = vec![Area::default(); n];
        for i in (0..n).rev() {
            let Some(demand) = pic_demand[i] else {
                continue;
            };
            let (reach, reads) = self.pictures[i];
            made[i] = demand.intersect(frame);
            read[i] = made[i].grown(reach, frame);
            if reads.pictures {
                for d in &mut pic_demand[..i] {
                    *d = union(*d, read[i]);
                }
            }
            if reads.planes {
                plane_demand = union(plane_demand, read[i]);
            }
            if reads.source {
                source = union(source, read[i]);
            }
        }
        let mut planes = Area::default();
        if let (Some(reach), Some(demand)) = (self.plane_reach, plane_demand) {
            planes = demand.intersect(frame);
            if self.plane_reads_source {
                source = union(source, planes.grown(reach, frame));
            }
        }
        Areas {
            source: source.map_or_else(Area::default, |s| s.intersect(frame)),
            planes,
            made,
            read,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pic(reach: usize) -> Declared {
        Declared {
            footprint: Footprint::Spatial {
                reach_px: reach,
                produces: Produces::Picture,
            },
            reads: Reads {
                source: true,
                planes: true,
                pictures: true,
            },
        }
    }

    fn plane(reach: usize) -> Declared {
        Declared {
            footprint: Footprint::Spatial {
                reach_px: reach,
                produces: Produces::Plane,
            },
            reads: Reads {
                source: true,
                ..Reads::default()
            },
        }
    }

    #[test]
    fn nothing_spatial_reads_the_need() {
        let s = Schedule::new([]);
        let frame = Area::whole(100, 80);
        let need = Area {
            x: 10,
            y: 10,
            w: 20,
            h: 20,
        };
        assert_eq!(s.infer(need, &frame).source, need);
    }

    #[test]
    fn the_sum_of_reaches_is_read_and_cut_to_the_frame() {
        let frame = Area::whole(100, 80);
        let s = Schedule::new([plane(4), pic(3), pic(5)]);
        let need = Area {
            x: 2,
            y: 30,
            w: 20,
            h: 20,
        };
        let a = s.infer(need, &frame);
        assert_eq!(a.source, need.grown(12, &frame));
        assert_eq!(a.made[1], need);
        assert_eq!(a.read[1], need.grown(5, &frame));
        assert_eq!(a.made[0], need.grown(5, &frame));
        assert_eq!(a.planes, need.grown(8, &frame));
        assert_eq!(s.total_reach(), 12);
    }

    #[test]
    fn a_step_with_no_consumer_reach_is_idempotent() {
        let frame = Area::whole(64, 64);
        let s = Schedule::new([plane(0)]);
        let need = Area::whole(10, 10);
        let a = s.infer(need, &frame);
        assert_eq!((a.source, a.planes), (need, need));
    }
}
