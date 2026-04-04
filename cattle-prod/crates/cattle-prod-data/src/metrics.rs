/// Structural quality scoring metrics for predicted protein structures.
///
/// All functions operate on `&[[f32; 3]]` coordinate arrays where each
/// element is an (x, y, z) position. An optional boolean mask selects which
/// atoms participate in the calculation.

/// Euclidean distance between two 3-D points.
#[inline]
fn dist(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Local Distance Difference Test (lDDT).
///
/// For every pair of atoms (i, j) within `inclusion_radius` in the *true*
/// structure, check whether the predicted distance deviates by less than each
/// of the four thresholds (0.5, 1.0, 2.0, 4.0 Å by default, scaled by
/// `cutoff`). The score is the fraction of preserved distances averaged over
/// all thresholds.
///
/// `mask`: if `Some`, only atoms where `mask[i]` is `true` are considered.
pub fn lddt(
    pred_coords: &[[f32; 3]],
    true_coords: &[[f32; 3]],
    mask: Option<&[bool]>,
    cutoff: f64,
) -> f64 {
    let n = pred_coords.len().min(true_coords.len());
    if n == 0 {
        return 0.0;
    }

    let inclusion_radius: f32 = 15.0;
    let thresholds: [f64; 4] = [0.5 * cutoff, 1.0 * cutoff, 2.0 * cutoff, 4.0 * cutoff];

    let is_active = |i: usize| -> bool {
        mask.map_or(true, |m| i < m.len() && m[i])
    };

    let mut total_preserved: f64 = 0.0;
    let mut total_pairs: u64 = 0;

    for i in 0..n {
        if !is_active(i) {
            continue;
        }
        for j in (i + 1)..n {
            if !is_active(j) {
                continue;
            }
            let d_true = dist(&true_coords[i], &true_coords[j]);
            if d_true > inclusion_radius {
                continue;
            }

            let d_pred = dist(&pred_coords[i], &pred_coords[j]);
            let delta = (d_pred as f64 - d_true as f64).abs();

            for &t in &thresholds {
                if delta < t {
                    total_preserved += 1.0;
                }
            }
            total_pairs += 1;
        }
    }

    if total_pairs == 0 {
        return 0.0;
    }

    total_preserved / (total_pairs as f64 * thresholds.len() as f64)
}

/// Root Mean Square Deviation between two coordinate sets.
///
/// `mask`: if `Some`, only atoms where `mask[i]` is `true` contribute.
pub fn rmsd(
    pred_coords: &[[f32; 3]],
    true_coords: &[[f32; 3]],
    mask: Option<&[bool]>,
) -> f64 {
    let n = pred_coords.len().min(true_coords.len());
    if n == 0 {
        return 0.0;
    }

    let mut sum_sq: f64 = 0.0;
    let mut count: u64 = 0;

    for i in 0..n {
        if let Some(m) = mask {
            if i >= m.len() || !m[i] {
                continue;
            }
        }
        let dx = (pred_coords[i][0] - true_coords[i][0]) as f64;
        let dy = (pred_coords[i][1] - true_coords[i][1]) as f64;
        let dz = (pred_coords[i][2] - true_coords[i][2]) as f64;
        sum_sq += dx * dx + dy * dy + dz * dz;
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }

    (sum_sq / count as f64).sqrt()
}

/// Global Distance Test – Total Score (GDT-TS).
///
/// Reports the average percentage of Cα atoms that fall within 1, 2, 4, and
/// 8 Å of the corresponding true position after optimal superposition.
///
/// Because full Kabsch superposition is non-trivial, this implementation
/// operates on *pre-aligned* coordinate arrays (the caller is responsible for
/// superposition). The returned value is in [0, 1].
pub fn gdt_ts(pred_coords: &[[f32; 3]], true_coords: &[[f32; 3]]) -> f64 {
    let n = pred_coords.len().min(true_coords.len());
    if n == 0 {
        return 0.0;
    }

    let thresholds: [f32; 4] = [1.0, 2.0, 4.0, 8.0];
    let mut counts = [0u64; 4];

    for i in 0..n {
        let d = dist(&pred_coords[i], &true_coords[i]);
        for (t_idx, &t) in thresholds.iter().enumerate() {
            if d <= t {
                counts[t_idx] += 1;
            }
        }
    }

    let total: f64 = counts.iter().map(|&c| c as f64 / n as f64).sum();
    total / thresholds.len() as f64
}

/// Steric clash score.
///
/// Counts the number of atom pairs whose distance is below
/// `(r_i + r_j) * threshold` where `r_i` and `r_j` are van der Waals radii.
/// The score is normalised per 1000 atoms (like MolProbity convention).
///
/// `vdw_radii` must be the same length as `coords`. A typical default radius
/// for an unresolved atom type is 1.7 Å.
pub fn clash_score(
    coords: &[[f32; 3]],
    vdw_radii: &[f32],
    threshold: f64,
) -> f64 {
    let n = coords.len();
    if n < 2 {
        return 0.0;
    }
    assert_eq!(
        vdw_radii.len(),
        n,
        "vdw_radii length must match coords length"
    );

    let threshold = threshold as f32;
    let mut clashes: u64 = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let allowed = (vdw_radii[i] + vdw_radii[j]) * threshold;
            let d = dist(&coords[i], &coords[j]);
            if d < allowed {
                clashes += 1;
            }
        }
    }

    (clashes as f64 / n as f64) * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsd_identical() {
        let coords = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let r = rmsd(&coords, &coords, None);
        assert!(r.abs() < 1e-9, "RMSD of identical coords should be 0, got {r}");
    }

    #[test]
    fn test_rmsd_known() {
        let pred = vec![[0.0, 0.0, 0.0]];
        let truth = vec![[1.0, 0.0, 0.0]];
        let r = rmsd(&pred, &truth, None);
        assert!((r - 1.0).abs() < 1e-6, "Expected RMSD=1.0, got {r}");
    }

    #[test]
    fn test_rmsd_with_mask() {
        let pred = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let truth = vec![[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let mask = vec![true, false];
        let r = rmsd(&pred, &truth, Some(&mask));
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lddt_perfect() {
        let coords = vec![
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
            [11.4, 0.0, 0.0],
        ];
        let score = lddt(&coords, &coords, None, 1.0);
        assert!(
            (score - 1.0).abs() < 1e-9,
            "lDDT of identical structures should be 1.0, got {score}"
        );
    }

    #[test]
    fn test_lddt_imperfect() {
        let true_coords = vec![
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
        ];
        let pred_coords = vec![
            [0.0, 0.0, 0.0],
            [3.8, 3.0, 0.0],
            [7.6, 0.0, 0.0],
        ];
        let score = lddt(&pred_coords, &true_coords, None, 1.0);
        assert!(score > 0.0 && score < 1.0, "Expected partial lDDT, got {score}");
    }

    #[test]
    fn test_gdt_ts_perfect() {
        let coords = vec![[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
        let score = gdt_ts(&coords, &coords);
        assert!((score - 1.0).abs() < 1e-9, "GDT-TS of identical coords should be 1.0, got {score}");
    }

    #[test]
    fn test_gdt_ts_far_apart() {
        let pred = vec![[0.0, 0.0, 0.0]];
        let truth = vec![[100.0, 0.0, 0.0]];
        let score = gdt_ts(&pred, &truth);
        assert!(score.abs() < 1e-9, "GDT-TS should be ~0 for distant atoms, got {score}");
    }

    #[test]
    fn test_clash_score_no_clashes() {
        let coords = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let radii = vec![1.7, 1.7];
        let score = clash_score(&coords, &radii, 0.75);
        assert!(score.abs() < 1e-9, "No clashes expected, got {score}");
    }

    #[test]
    fn test_clash_score_has_clashes() {
        let coords = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let radii = vec![1.7, 1.7];
        let score = clash_score(&coords, &radii, 0.75);
        assert!(score > 0.0, "Should detect clash, got {score}");
    }
}
