use std::collections::HashMap;

use ndarray::{s, Array1, Array2, Array3, Array4, Axis};

use cattle_prod_core::constants::{
    make_restype_pseudobeta_idx, make_restype_rigidgroup_dense_atom_idx,
};
use cattle_prod_core::residue::STD_RESIDUES_WITH_GAP;

const NUM_DENSE_ATOMS: usize = 24;
const DEFAULT_MAX_TEMPLATES: usize = 4;
#[allow(dead_code)]
const DAYS_BEFORE_QUERY_DATE: i64 = 60;

#[derive(Debug, Clone, Copy)]
pub struct DistogramConfig {
    pub min_bin: f32,
    pub max_bin: f32,
    pub num_bins: usize,
}

impl Default for DistogramConfig {
    fn default() -> Self {
        Self {
            min_bin: 3.25,
            max_bin: 50.75,
            num_bins: 39,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TemplateHit {
    pub index: usize,
    pub name: String,
    pub aligned_cols: usize,
    pub sum_probs: Option<f64>,
    pub query: String,
    pub hit_sequence: String,
    pub indices_query: Vec<i32>,
    pub indices_hit: Vec<i32>,
}

impl TemplateHit {
    pub fn query_to_hit_mapping(&self) -> HashMap<i32, i32> {
        let mut m = HashMap::new();
        for (&qi, &hi) in self.indices_query.iter().zip(self.indices_hit.iter()) {
            if qi != -1 && hi != -1 {
                m.insert(qi, hi);
            }
        }
        m
    }
}

#[derive(Debug, Clone)]
pub struct Templates {
    pub aatype: Array2<i32>,
    pub atom_positions: Array4<f32>,
    pub atom_mask: Array3<f32>,
}

impl Templates {
    pub fn num_templates(&self) -> usize {
        self.aatype.shape()[0]
    }

    pub fn num_residues(&self) -> usize {
        self.aatype.shape()[1]
    }

    pub fn as_cattle_prod_dict(&self) -> TemplateFeatureDict {
        let nt = self.num_templates();
        let nr = self.num_residues();
        let config = DistogramConfig::default();

        let pseudobeta_idx = make_restype_pseudobeta_idx();

        let mut pb_masks = Array3::<f32>::zeros((nt, nr, nr));
        let mut dgrams = Array4::<f32>::zeros((nt, nr, nr, config.num_bins));
        let mut unit_vectors = Array4::<f32>::zeros((nt, nr, nr, 3));
        let mut bb_masks = Array3::<f32>::zeros((nt, nr, nr));

        let rigidgroup_idx = make_restype_rigidgroup_dense_atom_idx();

        for t in 0..nt {
            let aa = self.aatype.slice(s![t, ..]);
            let mask_2 = self.atom_mask.slice(s![t, .., ..]);  // [nr, dense]
            let pos_3 = self.atom_positions.slice(s![t, .., .., ..]); // [nr, dense, 3]

            let (pb_pos, pb_mask) = pseudo_beta_fn(aa.as_slice().unwrap(), &pos_3, &mask_2, &pseudobeta_idx);
            for i in 0..nr {
                for j in 0..nr {
                    pb_masks[[t, i, j]] = pb_mask[i] * pb_mask[j];
                }
            }

            let dgram = dgram_from_positions(&pb_pos, &config);
            for i in 0..nr {
                for j in 0..nr {
                    let m2d = pb_masks[[t, i, j]];
                    for b in 0..config.num_bins {
                        dgrams[[t, i, j, b]] = dgram[[i, j, b]] * m2d;
                    }
                }
            }

            let (uv, bb_mask_2d) = compute_template_unit_vector(
                aa.as_slice().unwrap(),
                &pos_3,
                &mask_2,
                &rigidgroup_idx,
            );
            for i in 0..nr {
                for j in 0..nr {
                    bb_masks[[t, i, j]] = bb_mask_2d[[i, j]];
                    for k in 0..3 {
                        unit_vectors[[t, i, j, k]] = uv[[i, j, k]] * bb_mask_2d[[i, j]];
                    }
                }
            }
        }

        TemplateFeatureDict {
            aatype: self.aatype.clone(),
            atom_positions: self.atom_positions.clone(),
            atom_mask: self.atom_mask.clone(),
            pseudo_beta_mask: pb_masks,
            distogram: dgrams,
            unit_vector: unit_vectors,
            backbone_frame_mask: bb_masks,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TemplateFeatureDict {
    pub aatype: Array2<i32>,
    pub atom_positions: Array4<f32>,
    pub atom_mask: Array3<f32>,
    pub pseudo_beta_mask: Array3<f32>,
    pub distogram: Array4<f32>,
    pub unit_vector: Array4<f32>,
    pub backbone_frame_mask: Array3<f32>,
}

pub fn empty_template_features(num_res: usize) -> RawTemplateFeatures {
    let gap_id = *STD_RESIDUES_WITH_GAP.get("-").unwrap_or(&31) as i32;
    RawTemplateFeatures {
        aatype: Array2::from_elem((1, num_res), gap_id),
        atom_positions: Array4::<f32>::zeros((1, num_res, NUM_DENSE_ATOMS, 3)),
        atom_mask: Array3::<f32>::zeros((1, num_res, NUM_DENSE_ATOMS)),
    }
}

#[derive(Debug, Clone)]
pub struct RawTemplateFeatures {
    pub aatype: Array2<i32>,
    pub atom_positions: Array4<f32>,
    pub atom_mask: Array3<f32>,
}

impl RawTemplateFeatures {
    pub fn num_templates(&self) -> usize {
        self.aatype.shape()[0]
    }

    pub fn reduce(&self, max_templates: usize) -> Self {
        let nt = self.num_templates().min(max_templates);
        Self {
            aatype: self.aatype.slice(s![..nt, ..]).to_owned(),
            atom_positions: self.atom_positions.slice(s![..nt, .., .., ..]).to_owned(),
            atom_mask: self.atom_mask.slice(s![..nt, .., ..]).to_owned(),
        }
    }

    pub fn pad_to_templates(&self, max_templates: usize) -> Self {
        let nt = self.num_templates();
        if nt >= max_templates {
            return self.reduce(max_templates);
        }
        let nr = self.aatype.shape()[1];
        let mut aa = Array2::<i32>::zeros((max_templates, nr));
        let mut ap = Array4::<f32>::zeros((max_templates, nr, NUM_DENSE_ATOMS, 3));
        let mut am = Array3::<f32>::zeros((max_templates, nr, NUM_DENSE_ATOMS));
        aa.slice_mut(s![..nt, ..]).assign(&self.aatype);
        ap.slice_mut(s![..nt, .., .., ..]).assign(&self.atom_positions);
        am.slice_mut(s![..nt, .., ..]).assign(&self.atom_mask);
        Self {
            aatype: aa,
            atom_positions: ap,
            atom_mask: am,
        }
    }

    pub fn into_templates(self) -> Templates {
        Templates {
            aatype: self.aatype,
            atom_positions: self.atom_positions,
            atom_mask: self.atom_mask,
        }
    }
}

pub struct TemplateAssemblyLine {
    pub max_templates: usize,
}

impl Default for TemplateAssemblyLine {
    fn default() -> Self {
        Self {
            max_templates: DEFAULT_MAX_TEMPLATES,
        }
    }
}

impl TemplateAssemblyLine {
    pub fn new(max_templates: usize) -> Self {
        Self { max_templates }
    }

    pub fn assemble(
        &self,
        chain_features: Vec<RawTemplateFeatures>,
        standard_token_idxs: &[usize],
    ) -> Templates {
        let padded: Vec<RawTemplateFeatures> = chain_features
            .into_iter()
            .map(|f| f.pad_to_templates(self.max_templates))
            .collect();

        if padded.is_empty() {
            return Templates {
                aatype: Array2::zeros((self.max_templates, 0)),
                atom_positions: Array4::zeros((self.max_templates, 0, NUM_DENSE_ATOMS, 3)),
                atom_mask: Array3::zeros((self.max_templates, 0, NUM_DENSE_ATOMS)),
            };
        }

        let all_aa: Vec<_> = padded.iter().map(|p| p.aatype.view()).collect();
        let merged_aa = ndarray::concatenate(Axis(1), &all_aa).unwrap();

        let all_ap: Vec<_> = padded.iter().map(|p| p.atom_positions.view()).collect();
        let merged_ap = ndarray::concatenate(Axis(1), &all_ap).unwrap();

        let all_am: Vec<_> = padded.iter().map(|p| p.atom_mask.view()).collect();
        let merged_am = ndarray::concatenate(Axis(1), &all_am).unwrap();

        let aa = merged_aa.select(Axis(1), standard_token_idxs);
        let ap = merged_ap.select(Axis(1), standard_token_idxs);
        let am = merged_am.select(Axis(1), standard_token_idxs);

        Templates {
            aatype: aa,
            atom_positions: ap,
            atom_mask: am,
        }
    }
}

fn pseudo_beta_fn(
    aatype: &[i32],
    positions: &ndarray::ArrayView3<f32>,
    mask: &ndarray::ArrayView2<f32>,
    pseudobeta_idx: &[i32],
) -> (Array2<f32>, Array1<f32>) {
    let nr = aatype.len();
    let mut pb_pos = Array2::<f32>::zeros((nr, 3));
    let mut pb_mask = Array1::<f32>::zeros(nr);

    for i in 0..nr {
        let aa = aatype[i].max(0) as usize;
        let pb_idx = if aa < pseudobeta_idx.len() {
            pseudobeta_idx[aa] as usize
        } else {
            0
        };
        for k in 0..3 {
            pb_pos[[i, k]] = positions[[i, pb_idx, k]];
        }
        pb_mask[i] = mask[[i, pb_idx]];
    }
    (pb_pos, pb_mask)
}

fn dgram_from_positions(positions: &Array2<f32>, config: &DistogramConfig) -> Array3<f32> {
    let nr = positions.shape()[0];
    let nb = config.num_bins;

    let mut lower = Array1::<f32>::zeros(nb);
    let step = (config.max_bin - config.min_bin) / (nb as f32 - 1.0);
    for b in 0..nb {
        let edge = config.min_bin + b as f32 * step;
        lower[b] = edge * edge;
    }
    let mut upper = Array1::<f32>::zeros(nb);
    for b in 0..(nb - 1) {
        upper[b] = lower[b + 1];
    }
    upper[nb - 1] = 1e8;

    let mut dgram = Array3::<f32>::zeros((nr, nr, nb));
    for i in 0..nr {
        for j in 0..nr {
            let mut dist2 = 0.0f32;
            for k in 0..3 {
                let d = positions[[i, k]] - positions[[j, k]];
                dist2 += d * d;
            }
            for b in 0..nb {
                if dist2 > lower[b] && dist2 < upper[b] {
                    dgram[[i, j, b]] = 1.0;
                }
            }
        }
    }
    dgram
}

fn compute_template_unit_vector(
    aatype: &[i32],
    positions: &ndarray::ArrayView3<f32>,
    mask: &ndarray::ArrayView2<f32>,
    rigidgroup_idx: &ndarray::Array3<i32>,
) -> (Array3<f32>, Array2<f32>) {
    let nr = aatype.len();
    let eps = 1e-6f32;

    let mut ca_pos = Array2::<f32>::zeros((nr, 3));
    let mut bb_mask = Array1::<f32>::zeros(nr);

    let mut e1 = Array2::<f32>::zeros((nr, 3));
    let mut e2 = Array2::<f32>::zeros((nr, 3));
    let mut e3 = Array2::<f32>::zeros((nr, 3));

    for i in 0..nr {
        let aa = aatype[i].max(0) as usize;
        let aa_clamped = aa.min(rigidgroup_idx.shape()[0] - 1);

        let c_idx = rigidgroup_idx[[aa_clamped, 0, 0]] as usize;
        let ca_idx = rigidgroup_idx[[aa_clamped, 0, 1]] as usize;
        let n_idx = rigidgroup_idx[[aa_clamped, 0, 2]] as usize;

        let c_mask = mask[[i, c_idx]];
        let ca_mask = mask[[i, ca_idx]];
        let n_mask = mask[[i, n_idx]];
        bb_mask[i] = c_mask * ca_mask * n_mask;

        let mut c_p = [0.0f32; 3];
        let mut ca_p = [0.0f32; 3];
        let mut n_p = [0.0f32; 3];
        for k in 0..3 {
            c_p[k] = positions[[i, c_idx, k]];
            ca_p[k] = positions[[i, ca_idx, k]];
            n_p[k] = positions[[i, n_idx, k]];
            ca_pos[[i, k]] = ca_p[k];
        }

        let mut v1 = [0.0f32; 3];
        let mut v2 = [0.0f32; 3];
        for k in 0..3 {
            v1[k] = c_p[k] - ca_p[k];
            v2[k] = n_p[k] - ca_p[k];
        }
        let v1_norm = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt() + eps;
        for k in 0..3 {
            e1[[i, k]] = v1[k] / v1_norm;
        }

        let dot_v2_e1 = v2[0] * e1[[i, 0]] + v2[1] * e1[[i, 1]] + v2[2] * e1[[i, 2]];
        let mut u2 = [0.0f32; 3];
        for k in 0..3 {
            u2[k] = v2[k] - dot_v2_e1 * e1[[i, k]];
        }
        let u2_norm = (u2[0] * u2[0] + u2[1] * u2[1] + u2[2] * u2[2]).sqrt() + eps;
        for k in 0..3 {
            e2[[i, k]] = u2[k] / u2_norm;
        }

        e3[[i, 0]] = e1[[i, 1]] * e2[[i, 2]] - e1[[i, 2]] * e2[[i, 1]];
        e3[[i, 1]] = e1[[i, 2]] * e2[[i, 0]] - e1[[i, 0]] * e2[[i, 2]];
        e3[[i, 2]] = e1[[i, 0]] * e2[[i, 1]] - e1[[i, 1]] * e2[[i, 0]];
    }

    let mut unit_vector = Array3::<f32>::zeros((nr, nr, 3));
    let mut mask_2d = Array2::<f32>::zeros((nr, nr));

    for i in 0..nr {
        for j in 0..nr {
            mask_2d[[i, j]] = bb_mask[i] * bb_mask[j];
            let mut diff = [0.0f32; 3];
            for k in 0..3 {
                diff[k] = ca_pos[[j, k]] - ca_pos[[i, k]];
            }
            for k in 0..3 {
                let r_col_k = [e1[[i, k]], e2[[i, k]], e3[[i, k]]];
                unit_vector[[i, j, k]] =
                    r_col_k[0] * diff[0] + r_col_k[1] * diff[1] + r_col_k[2] * diff[2];
            }
        }
    }

    (unit_vector, mask_2d)
}

pub fn parse_hhr(hhr_string: &str) -> Vec<TemplateHit> {
    let lines: Vec<&str> = hhr_string.lines().collect();
    let block_starts: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("No "))
        .map(|(i, _)| i)
        .collect();

    if block_starts.is_empty() {
        return Vec::new();
    }

    let mut hits = Vec::new();
    for w in block_starts.windows(2) {
        if let Some(h) = parse_hhr_block(&lines[w[0]..w[1]]) {
            hits.push(h);
        }
    }
    if let Some(h) = parse_hhr_block(&lines[*block_starts.last().unwrap()..]) {
        hits.push(h);
    }
    hits
}

fn parse_hhr_block(lines: &[&str]) -> Option<TemplateHit> {
    if lines.is_empty() {
        return None;
    }

    let hit_num: usize = lines[0].split_whitespace().last()?.parse().ok()?;
    let hit_name = if lines.len() > 1 {
        lines[1].trim_start_matches('>').trim().to_string()
    } else {
        String::new()
    };

    let mut aligned_cols = 0usize;
    let mut sum_probs = None;
    if lines.len() > 2 {
        let summary = lines[2];
        if let Some(ac_start) = summary.find("Aligned_cols=") {
            let rest = &summary[ac_start + 13..];
            if let Some(num_end) = rest.find(|c: char| !c.is_ascii_digit()) {
                aligned_cols = rest[..num_end].parse().unwrap_or(0);
            }
        }
        if let Some(sp_start) = summary.find("Sum_probs=") {
            let rest = &summary[sp_start + 10..];
            if let Some(num_end) = rest.find(|c: char| !c.is_ascii_digit() && c != '.') {
                sum_probs = rest[..num_end].parse().ok();
            } else {
                sum_probs = rest.trim().parse().ok();
            }
        }
    }

    let mut query = String::new();
    let mut hit_sequence = String::new();
    let mut idx_q = Vec::new();
    let mut idx_h = Vec::new();

    let re_q = regex_lite::Regex::new(r"\s+(\d+)\s+([A-Z-]+)\s+(\d+)").ok()?;
    let re_h = regex_lite::Regex::new(r"\s+(\d+)\s+([A-Z-]+)").ok()?;

    for line in &lines[3..] {
        if line.starts_with("Q ") && !line.starts_with("Q ss_") && !line.starts_with("Q Consensus") {
            let after17 = if line.len() > 17 { &line[17..] } else { "" };
            if let Some(cap) = re_q.captures(after17) {
                let start: i32 = cap[1].parse::<i32>().unwrap_or(0) - 1;
                let seq = &cap[2];
                query.push_str(seq);
                update_residue_indices(seq, start, &mut idx_q);
            }
        } else if line.starts_with("T ") && !line.starts_with("T ss_") && !line.starts_with("T Consensus") {
            let after17 = if line.len() > 17 { &line[17..] } else { "" };
            if let Some(cap) = re_h.captures(after17) {
                let start: i32 = cap[1].parse::<i32>().unwrap_or(0) - 1;
                let seq = &cap[2];
                hit_sequence.push_str(seq);
                update_residue_indices(seq, start, &mut idx_h);
            }
        }
    }

    Some(TemplateHit {
        index: hit_num,
        name: hit_name,
        aligned_cols,
        sum_probs,
        query,
        hit_sequence,
        indices_query: idx_q,
        indices_hit: idx_h,
    })
}

fn update_residue_indices(seq: &str, start: i32, indices: &mut Vec<i32>) {
    let mut curr = start;
    for ch in seq.chars() {
        if ch == '-' {
            indices.push(-1);
        } else {
            indices.push(curr);
            curr += 1;
        }
    }
}

pub fn parse_a3m_for_templates(query_seq: &str, a3m_str: &str, skip_first: bool) -> Vec<TemplateHit> {
    let mut seqs = Vec::new();
    let mut descs = Vec::new();
    let mut current_desc = String::new();
    let mut current_seq = String::new();

    for line in a3m_str.lines() {
        if line.starts_with('>') {
            if !current_desc.is_empty() || !current_seq.is_empty() {
                descs.push(current_desc);
                seqs.push(current_seq);
            }
            current_desc = line.to_string();
            current_seq = String::new();
        } else {
            current_seq.push_str(line.trim());
        }
    }
    if !current_desc.is_empty() || !current_seq.is_empty() {
        descs.push(current_desc);
        seqs.push(current_seq);
    }

    let start_idx = if skip_first { 1 } else { 0 };
    let idx_q = get_a3m_indices(query_seq, 0);

    let mut hits = Vec::new();
    for (i, (seq, desc)) in seqs.iter().zip(descs.iter()).enumerate().skip(start_idx) {
        if !desc.contains("mol:protein") {
            continue;
        }
        let meta = match parse_a3m_description(desc) {
            Some(m) => m,
            None => continue,
        };
        let cols = seq.chars().filter(|c| c.is_uppercase() && *c != '-').count();
        let idx_h = get_a3m_indices(seq, meta.start.saturating_sub(1) as i32);
        hits.push(TemplateHit {
            index: i,
            name: format!("{}_{}", meta.pdb_id, meta.chain),
            aligned_cols: cols,
            sum_probs: None,
            query: query_seq.to_string(),
            hit_sequence: seq.to_uppercase(),
            indices_query: idx_q.clone(),
            indices_hit: idx_h,
        });
    }
    hits
}

struct A3mMeta {
    pdb_id: String,
    chain: String,
    start: usize,
    #[allow(dead_code)]
    end: usize,
    #[allow(dead_code)]
    length: usize,
}

fn parse_a3m_description(desc: &str) -> Option<A3mMeta> {
    let desc = desc.trim().trim_start_matches('>');
    let re = regex_lite::Regex::new(
        r"^([a-zA-Z\d]{4})_(\w+)/([0-9]+)-([0-9]+).*protein length:([0-9]+)"
    ).ok()?;
    let cap = re.captures(desc)?;
    Some(A3mMeta {
        pdb_id: cap[1].to_string(),
        chain: cap[2].to_string(),
        start: cap[3].parse().unwrap_or(1),
        end: cap[4].parse().unwrap_or(0),
        length: cap[5].parse().unwrap_or(0),
    })
}

fn get_a3m_indices(seq: &str, start: i32) -> Vec<i32> {
    let mut indices = Vec::with_capacity(seq.len());
    let mut counter = start;
    for ch in seq.chars() {
        if ch == '-' {
            indices.push(-1);
            // gap
        } else if ch.is_lowercase() {
            counter += 1;
            // insertion: skip in output
        } else {
            indices.push(counter);
            counter += 1;
        }
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_template_features() {
        let feats = empty_template_features(10);
        assert_eq!(feats.aatype.shape(), &[1, 10]);
        assert_eq!(feats.atom_positions.shape(), &[1, 10, 24, 3]);
        assert_eq!(feats.atom_mask.shape(), &[1, 10, 24]);
    }

    #[test]
    fn test_pad_and_reduce() {
        let feats = empty_template_features(5);
        let padded = feats.pad_to_templates(4);
        assert_eq!(padded.aatype.shape(), &[4, 5]);

        let reduced = padded.reduce(2);
        assert_eq!(reduced.aatype.shape(), &[2, 5]);
    }

    #[test]
    fn test_assembly_line() {
        let f1 = empty_template_features(3);
        let f2 = empty_template_features(5);
        let al = TemplateAssemblyLine::new(4);
        let indices: Vec<usize> = (0..8).collect();
        let templates = al.assemble(vec![f1, f2], &indices);
        assert_eq!(templates.num_templates(), 4);
        assert_eq!(templates.num_residues(), 8);
    }

    #[test]
    fn test_distogram_config_default() {
        let cfg = DistogramConfig::default();
        assert_eq!(cfg.num_bins, 39);
        assert!((cfg.min_bin - 3.25).abs() < 1e-6);
    }

    #[test]
    fn test_dgram_zeros() {
        let pos = Array2::<f32>::zeros((3, 3));
        let config = DistogramConfig::default();
        let dgram = dgram_from_positions(&pos, &config);
        assert_eq!(dgram.shape(), &[3, 3, 39]);
    }

    #[test]
    fn test_parse_hhr_empty() {
        let hits = parse_hhr("");
        assert!(hits.is_empty());
    }

    #[test]
    fn test_update_residue_indices() {
        let mut indices = Vec::new();
        update_residue_indices("AV-G", 0, &mut indices);
        assert_eq!(indices, vec![0, 1, -1, 2]);
    }

    #[test]
    fn test_get_a3m_indices() {
        let idx = get_a3m_indices("AVaG", 0);
        assert_eq!(idx, vec![0, 1, 3]);
    }

    #[test]
    fn test_parse_a3m_empty() {
        let hits = parse_a3m_for_templates("ACGT", "", false);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_template_hit_mapping() {
        let hit = TemplateHit {
            index: 0,
            name: "test".to_string(),
            aligned_cols: 3,
            sum_probs: None,
            query: "ABC".to_string(),
            hit_sequence: "DEF".to_string(),
            indices_query: vec![0, 1, -1, 2],
            indices_hit: vec![0, -1, 1, 2],
        };
        let m = hit.query_to_hit_mapping();
        assert_eq!(m.get(&0), Some(&0));
        assert_eq!(m.get(&2), Some(&2));
        assert_eq!(m.get(&1), None);
    }
}
