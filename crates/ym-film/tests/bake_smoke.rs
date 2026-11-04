mod bake_common;
use bake_common::*;
use ym_color::display::cine::CineRendering;
use ym_film::Output;
use ym_film::bake::*;

#[test]
fn t3b_regions() {
    let out = Output::DisplayP3;
    let cine = CineRendering::new(out.target());
    let space = output_space(out);
    let c = chain("kodak_portra_400", out);
    let b = baker(4);
    let f = b.bake(&c, &BakeSettings { t3b_size: 65, ..BakeSettings::of(&c) }).unwrap();
    let hi = f.t3b().axis_max();
    eprintln!("hi {hi:?}");
    let mut rng = Rng(3);
    // buckets by luminance
    let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); 8];
    let mut unreach = Vec::new();
    let mut reach = Vec::new();
    for _ in 0..40000 {
        let d = [hi[0]*rng.unit(), hi[1]*rng.unit(), hi[2]*rng.unit()];
        let disp = c.scan_stage().display(d);
        let inv = cine.inverse(disp);
        let exact = ym_color::logc::logc4_encode_rgb(inv.scene.0);
        let q = f.t3b().sample(d);
        let ca = codes10(&cine, space, exact);
        let cq = codes10(&cine, space, q);
        let e = (0..3).map(|k| (ca[k]-cq[k]).abs()).fold(0.0, f64::max);
        let y = c.scan_stage().luminance(disp);
        let bi = if y <= 0.0 { 0 } else { ((-y.log10()*2.0).floor().max(0.0) as usize).min(7) };
        buckets[bi].push(e);
        if inv.chroma_fraction < 1.0 { unreach.push(e) } else { reach.push(e) }
    }
    for (i, mut v) in buckets.into_iter().enumerate() {
        if v.is_empty() { continue; }
        v.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let n = v.len();
        eprintln!("Y in 10^-{:.1}..: n {n} max {:.2} p99 {:.2} med {:.3}", i as f64/2.0, v[n-1], v[n*99/100], v[n/2]);
    }
    for (nm, mut v) in [("unreachable", unreach), ("reachable", reach)] {
        if v.is_empty() { continue; }
        v.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let n = v.len();
        eprintln!("{nm}: n {n} max {:.2} p99 {:.2} med {:.3}", v[n-1], v[n*99/100], v[n/2]);
    }
}
