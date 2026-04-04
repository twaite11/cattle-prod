use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, IxDyn};

use cattle_prod_core::constants::ELEMENT_SYMBOLS;
use cattle_prod_core::residue::NUM_STD_RESIDUES;
use cattle_prod_core::token::TokenArray;

use crate::atom_array::AtomArray;

const NUM_ELEMENT_CLASSES: usize = 128;
const ATOM_NAME_MAX_CHARS: usize = 4;
const ATOM_NAME_NUM_CLASSES: usize = 64;

pub struct Featurizer {
    pub max_atoms_per_token: usize,
}

impl Featurizer {
    pub fn new(max_atoms_per_token: usize) -> Self {
        Self { max_atoms_per_token }
    }

    /// 32-class one-hot encoding of token residue-type values.
    /// Tokens with value >= NUM_STD_RESIDUES are clamped to the last class.
    pub fn get_restype_onehot(&self, token_array: &TokenArray) -> Array2<f32> {
        let n = token_array.len();
        let nc = NUM_STD_RESIDUES;
        let mut out = Array2::<f32>::zeros((n, nc));

        for (i, tok) in token_array.iter().enumerate() {
            let cls = (tok.value as usize).min(nc - 1);
            out[[i, cls]] = 1.0;
        }
        out
    }

    /// N_token × N_token adjacency matrix: entry (i,j) = 1 when any atom
    /// belonging to token i shares a bond with any atom belonging to token j.
    pub fn get_token_bonds(
        &self,
        token_array: &TokenArray,
        atom_array: &AtomArray,
    ) -> Array2<f32> {
        let n = token_array.len();
        let mut adj = Array2::<f32>::zeros((n, n));

        // atom-index → token-index lookup
        let atom_to_tok = self.build_atom_to_token(token_array, atom_array.len());

        for bond in &atom_array.bonds {
            let ti = atom_to_tok[bond.atom_i];
            let tj = atom_to_tok[bond.atom_j];
            if ti >= 0 && tj >= 0 {
                let ti = ti as usize;
                let tj = tj as usize;
                if ti < n && tj < n {
                    adj[[ti, tj]] = 1.0;
                    adj[[tj, ti]] = 1.0;
                }
            }
        }
        adj
    }

    /// Compute per-token reference features packed to `max_atoms_per_token`.
    ///
    /// Returned keys:
    /// - `ref_pos`              : `[N_tok, max_atoms, 3]`
    /// - `ref_mask`             : `[N_tok, max_atoms]`
    /// - `ref_element`          : `[N_tok, max_atoms, 128]`
    /// - `ref_charge`           : `[N_tok, max_atoms]`
    /// - `ref_atom_name_chars`  : `[N_tok, max_atoms, 4, 64]`
    /// - `ref_space_uid`        : `[N_tok, max_atoms]`
    pub fn get_reference_features(
        &self,
        token_array: &TokenArray,
        atom_array: &AtomArray,
    ) -> HashMap<String, ArrayD<f32>> {
        let nt = token_array.len();
        let ma = self.max_atoms_per_token;

        let mut ref_pos =
            ArrayD::<f32>::zeros(IxDyn(&[nt, ma, 3]));
        let mut ref_mask =
            ArrayD::<f32>::zeros(IxDyn(&[nt, ma]));
        let mut ref_element =
            ArrayD::<f32>::zeros(IxDyn(&[nt, ma, NUM_ELEMENT_CLASSES]));
        let ref_charge =
            ArrayD::<f32>::zeros(IxDyn(&[nt, ma]));
        let mut ref_atom_name_chars =
            ArrayD::<f32>::zeros(IxDyn(&[nt, ma, ATOM_NAME_MAX_CHARS, ATOM_NAME_NUM_CLASSES]));
        let mut ref_space_uid =
            ArrayD::<f32>::zeros(IxDyn(&[nt, ma]));

        for (t, tok) in token_array.iter().enumerate() {
            let n_atoms = tok.atom_indices.len().min(ma);
            for a in 0..n_atoms {
                let ai = tok.atom_indices[a];
                if ai >= atom_array.len() {
                    continue;
                }

                ref_pos[IxDyn(&[t, a, 0])] = atom_array.coord[ai][0];
                ref_pos[IxDyn(&[t, a, 1])] = atom_array.coord[ai][1];
                ref_pos[IxDyn(&[t, a, 2])] = atom_array.coord[ai][2];
                ref_mask[IxDyn(&[t, a])] = 1.0;

                // element one-hot
                let elem_idx = element_onehot_index(&atom_array.element[ai]);
                if elem_idx < NUM_ELEMENT_CLASSES {
                    ref_element[IxDyn(&[t, a, elem_idx])] = 1.0;
                }

                // charge is unavailable in mmCIF atom_site – leave as 0

                // atom name character encoding
                encode_atom_name(
                    &atom_array.atom_name[ai],
                    &mut ref_atom_name_chars,
                    t,
                    a,
                );

                ref_space_uid[IxDyn(&[t, a])] = t as f32;
            }
        }

        let mut m = HashMap::new();
        m.insert("ref_pos".into(), ref_pos);
        m.insert("ref_mask".into(), ref_mask);
        m.insert("ref_element".into(), ref_element);
        m.insert("ref_charge".into(), ref_charge);
        m.insert("ref_atom_name_chars".into(), ref_atom_name_chars);
        m.insert("ref_space_uid".into(), ref_space_uid);
        m
    }

    /// Map every atom to the token index that owns it. Returns `[N_atoms]`
    /// with -1 for atoms not claimed by any token.
    pub fn get_atom_to_token_idx(
        &self,
        token_array: &TokenArray,
        n_atoms: usize,
    ) -> Array1<i64> {
        let v = self.build_atom_to_token(token_array, n_atoms);
        Array1::from_vec(v)
    }

    /// Convenience: run all feature computations and return a combined dict.
    pub fn featurize(
        &self,
        token_array: &TokenArray,
        atom_array: &AtomArray,
    ) -> HashMap<String, ArrayD<f32>> {
        let mut features = HashMap::new();

        // restype one-hot [N_tok, 32]
        let onehot = self.get_restype_onehot(token_array);
        features.insert(
            "restype".into(),
            onehot.into_dyn().into_dimensionality().unwrap(),
        );

        // token bonds [N_tok, N_tok]
        let bonds = self.get_token_bonds(token_array, atom_array);
        features.insert(
            "token_bonds".into(),
            bonds.into_dyn().into_dimensionality().unwrap(),
        );

        // reference features
        let ref_feats = self.get_reference_features(token_array, atom_array);
        features.extend(ref_feats);

        // atom-to-token [N_atoms] (stored as f32 for uniformity)
        let a2t = self.build_atom_to_token(token_array, atom_array.len());
        let a2t_f: Vec<f32> = a2t.into_iter().map(|v| v as f32).collect();
        let a2t_arr = Array1::from_vec(a2t_f);
        features.insert(
            "atom_to_token_idx".into(),
            a2t_arr.into_dyn().into_dimensionality().unwrap(),
        );

        // residue_index [N_tok] – sequential index per token
        let res_idx: Vec<f32> = (0..token_array.len()).map(|i| i as f32).collect();
        features.insert(
            "residue_index".into(),
            Array1::from_vec(res_idx)
                .into_dyn()
                .into_dimensionality()
                .unwrap(),
        );

        features
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn build_atom_to_token(&self, token_array: &TokenArray, n_atoms: usize) -> Vec<i64> {
        let mut mapping = vec![-1i64; n_atoms];
        for (t, tok) in token_array.iter().enumerate() {
            for &ai in &tok.atom_indices {
                if ai < n_atoms {
                    mapping[ai] = t as i64;
                }
            }
        }
        mapping
    }
}

// ---------------------------------------------------------------------------
// Element helpers
// ---------------------------------------------------------------------------

fn element_onehot_index(symbol: &str) -> usize {
    let upper = symbol.to_uppercase();
    ELEMENT_SYMBOLS
        .iter()
        .position(|&s| s == upper.as_str())
        .unwrap_or(NUM_ELEMENT_CLASSES - 1)
}

fn encode_atom_name(
    name: &str,
    arr: &mut ArrayD<f32>,
    tok_idx: usize,
    atom_idx: usize,
) {
    for (c_pos, ch) in name.chars().take(ATOM_NAME_MAX_CHARS).enumerate() {
        let cls = (ch as usize) % ATOM_NAME_NUM_CLASSES;
        arr[IxDyn(&[tok_idx, atom_idx, c_pos, cls])] = 1.0;
    }
}
