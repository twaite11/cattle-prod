use std::collections::HashMap;

use anyhow::{Context, Result};
use ndarray::ArrayD;
use serde::Deserialize;

use cattle_prod_core::constants::RES_ATOMS_DICT;
use cattle_prod_core::residue::{
    DNA_STD_RESIDUES, PROT_STD_RESIDUES_ONE_TO_THREE, RNA_STD_RESIDUES,
};

use crate::atom_array::AtomArray;
use crate::featurizer::Featurizer;
use crate::tokenizer::AtomArrayTokenizer;

// ---------------------------------------------------------------------------
// JSON schema for inference input
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceInput {
    pub sequences: Vec<SequenceEntry>,
    #[serde(default)]
    pub name: String,
    #[serde(default = "default_model_seeds")]
    #[serde(rename = "modelSeeds")]
    pub model_seeds: Vec<u64>,
}

fn default_model_seeds() -> Vec<u64> {
    vec![101]
}

/// Each entry in the `sequences` array is exactly one of the supported entity
/// types. We use `serde(untagged)` so the deserializer tries each variant in
/// order.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SequenceEntry {
    Protein {
        #[serde(rename = "proteinChain")]
        protein_chain: ProteinChainDef,
    },
    Rna {
        #[serde(rename = "rnaSequence")]
        rna_sequence: NucleotideChainDef,
    },
    Dna {
        #[serde(rename = "dnaSequence")]
        dna_sequence: NucleotideChainDef,
    },
    Ligand {
        ligand: LigandDef,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProteinChainDef {
    pub sequence: String,
    #[serde(default = "one")]
    pub count: usize,
    #[serde(default)]
    pub modifications: Vec<ModificationDef>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NucleotideChainDef {
    pub sequence: String,
    #[serde(default = "one")]
    pub count: usize,
    #[serde(default)]
    pub modifications: Vec<ModificationDef>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LigandDef {
    /// CCD code (Chemical Component Dictionary identifier such as "ATP", "HEM").
    #[serde(default, rename = "CCD")]
    pub ccd: Option<String>,
    /// SMILES string as an alternative ligand specification.
    #[serde(default)]
    pub smiles: Option<String>,
    #[serde(default = "one")]
    pub count: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModificationDef {
    #[serde(rename = "ptmType")]
    pub ptm_type: String,
    #[serde(rename = "ptmPosition")]
    pub ptm_position: usize,
}

fn one() -> usize {
    1
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Deserialize the inference JSON file (an array of `InferenceInput`).
pub fn parse_inference_json(path: &str) -> Result<Vec<InferenceInput>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read inference JSON: {}", path))?;
    let inputs: Vec<InferenceInput> = serde_json::from_str(&content)
        .with_context(|| "failed to parse inference JSON")?;
    Ok(inputs)
}

/// Convenience: parse from a string slice.
pub fn parse_inference_json_str(json: &str) -> Result<Vec<InferenceInput>> {
    let inputs: Vec<InferenceInput> = serde_json::from_str(json)?;
    Ok(inputs)
}

// ---------------------------------------------------------------------------
// AtomArray construction from inference entities
// ---------------------------------------------------------------------------

/// Build an `AtomArray` from a single `InferenceInput` by expanding each
/// entity definition into atoms.
pub struct SampleDictToFeatures {
    pub max_atoms_per_token: usize,
}

impl SampleDictToFeatures {
    pub fn new(max_atoms_per_token: usize) -> Self {
        Self { max_atoms_per_token }
    }

    /// Convert the parsed JSON entities into an `AtomArray`.
    ///
    /// Coordinates are set to zero – this is inference input, actual
    /// positions will be predicted by the model.
    pub fn build_atom_array(&self, input: &InferenceInput) -> AtomArray {
        let mut array = AtomArray::new();
        let mut chain_counter = 0u32;
        let mut entity_counter = 1i32;

        for entry in &input.sequences {
            match entry {
                SequenceEntry::Protein { protein_chain } => {
                    for _ in 0..protein_chain.count {
                        let cid = chain_label(chain_counter);
                        chain_counter += 1;
                        self.expand_protein(
                            &protein_chain.sequence,
                            &cid,
                            entity_counter,
                            &mut array,
                        );
                    }
                    entity_counter += 1;
                }
                SequenceEntry::Rna { rna_sequence } => {
                    for _ in 0..rna_sequence.count {
                        let cid = chain_label(chain_counter);
                        chain_counter += 1;
                        self.expand_rna(
                            &rna_sequence.sequence,
                            &cid,
                            entity_counter,
                            &mut array,
                        );
                    }
                    entity_counter += 1;
                }
                SequenceEntry::Dna { dna_sequence } => {
                    for _ in 0..dna_sequence.count {
                        let cid = chain_label(chain_counter);
                        chain_counter += 1;
                        self.expand_dna(
                            &dna_sequence.sequence,
                            &cid,
                            entity_counter,
                            &mut array,
                        );
                    }
                    entity_counter += 1;
                }
                SequenceEntry::Ligand { ligand } => {
                    for _ in 0..ligand.count {
                        let cid = chain_label(chain_counter);
                        chain_counter += 1;
                        self.expand_ligand(ligand, &cid, entity_counter, &mut array);
                    }
                    entity_counter += 1;
                }
            }
        }
        array
    }

    /// Full pipeline: JSON entities → AtomArray → TokenArray → feature dict.
    pub fn process(
        &self,
        input: &InferenceInput,
    ) -> Result<HashMap<String, ArrayD<f32>>> {
        let array = self.build_atom_array(input);
        let tokenizer = AtomArrayTokenizer::new(&array);
        let token_array = tokenizer.get_token_array();
        let featurizer = Featurizer::new(self.max_atoms_per_token);
        let features = featurizer.featurize(&token_array, &array);
        Ok(features)
    }

    // ------------------------------------------------------------------
    // Entity expansion helpers
    // ------------------------------------------------------------------

    fn expand_protein(
        &self,
        sequence: &str,
        chain_id: &str,
        entity_id: i32,
        array: &mut AtomArray,
    ) {
        for (res_idx, ch) in sequence.chars().enumerate() {
            let three = PROT_STD_RESIDUES_ONE_TO_THREE
                .get(&ch)
                .copied()
                .unwrap_or("UNK");

            let atom_names = residue_atom_names(three);
            for name in &atom_names {
                let elem = element_from_atom_name(name);
                array.push_atom(
                    name.to_string(),
                    elem.to_string(),
                    three.to_string(),
                    (res_idx + 1) as i32,
                    chain_id.to_string(),
                    [0.0, 0.0, 0.0],
                    0.0,
                    1.0,
                    "protein".to_string(),
                    entity_id,
                );
            }
        }
    }

    fn expand_rna(
        &self,
        sequence: &str,
        chain_id: &str,
        entity_id: i32,
        array: &mut AtomArray,
    ) {
        for (res_idx, ch) in sequence.chars().enumerate() {
            let resname = &ch.to_string();
            let key = if RNA_STD_RESIDUES.contains_key(resname.as_str()) {
                resname.as_str()
            } else {
                "N"
            };

            let atom_names = residue_atom_names(key);
            for name in &atom_names {
                let elem = element_from_atom_name(name);
                array.push_atom(
                    name.to_string(),
                    elem.to_string(),
                    key.to_string(),
                    (res_idx + 1) as i32,
                    chain_id.to_string(),
                    [0.0, 0.0, 0.0],
                    0.0,
                    1.0,
                    "rna".to_string(),
                    entity_id,
                );
            }
        }
    }

    fn expand_dna(
        &self,
        sequence: &str,
        chain_id: &str,
        entity_id: i32,
        array: &mut AtomArray,
    ) {
        for (res_idx, ch) in sequence.chars().enumerate() {
            let resname = format!("D{}", ch);
            let key = if DNA_STD_RESIDUES.contains_key(resname.as_str()) {
                resname
            } else {
                "DN".to_string()
            };

            let atom_names = residue_atom_names(&key);
            for name in &atom_names {
                let elem = element_from_atom_name(name);
                array.push_atom(
                    name.to_string(),
                    elem.to_string(),
                    key.clone(),
                    (res_idx + 1) as i32,
                    chain_id.to_string(),
                    [0.0, 0.0, 0.0],
                    0.0,
                    1.0,
                    "dna".to_string(),
                    entity_id,
                );
            }
        }
    }

    fn expand_ligand(
        &self,
        ligand: &LigandDef,
        chain_id: &str,
        entity_id: i32,
        array: &mut AtomArray,
    ) {
        let comp_id = ligand
            .ccd
            .as_deref()
            .unwrap_or("UNL");

        // Without a full CCD dictionary we represent the ligand as a single
        // carbon placeholder atom. A production system would look up the CCD
        // to get the real atom list.
        let atom_names = residue_atom_names(comp_id);
        if atom_names.is_empty() {
            array.push_atom(
                "C1".to_string(),
                "C".to_string(),
                comp_id.to_string(),
                1,
                chain_id.to_string(),
                [0.0, 0.0, 0.0],
                0.0,
                1.0,
                "ligand".to_string(),
                entity_id,
            );
        } else {
            for name in &atom_names {
                let elem = element_from_atom_name(name);
                array.push_atom(
                    name.to_string(),
                    elem.to_string(),
                    comp_id.to_string(),
                    1,
                    chain_id.to_string(),
                    [0.0, 0.0, 0.0],
                    0.0,
                    1.0,
                    "ligand".to_string(),
                    entity_id,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Produce chain label strings: A, B, ... Z, AA, AB, ...
fn chain_label(index: u32) -> String {
    if index < 26 {
        return String::from((b'A' + index as u8) as char);
    }
    let first = (index / 26 - 1) as u8;
    let second = (index % 26) as u8;
    format!("{}{}", (b'A' + first) as char, (b'A' + second) as char)
}

/// Look up known atom names for a residue from `RES_ATOMS_DICT`.
fn residue_atom_names(res_name: &str) -> Vec<String> {
    match RES_ATOMS_DICT.get(res_name) {
        Some(map) => {
            let mut pairs: Vec<(&str, usize)> = map.iter().map(|(&k, &v)| (k, v)).collect();
            pairs.sort_by_key(|&(_, idx)| idx);
            pairs.into_iter().map(|(n, _)| n.to_string()).collect()
        }
        None => Vec::new(),
    }
}

/// Heuristic: derive element symbol from atom name.
/// Standard PDB convention: leading character(s) before any digit or prime.
fn element_from_atom_name(name: &str) -> &'static str {
    let name = name.trim();
    if name.is_empty() {
        return "C";
    }
    // single-character element names
    match name.chars().next().unwrap() {
        'C' => "C",
        'N' => "N",
        'O' => "O",
        'S' => "S",
        'P' => "P",
        'H' => "H",
        'F' => "F",
        _ => "C",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_inference_json() {
        let json = r#"[{
            "sequences": [
                {"proteinChain": {"sequence": "ACDE", "count": 1}},
                {"ligand": {"CCD": "ATP"}}
            ],
            "name": "test"
        }]"#;
        let inputs = parse_inference_json_str(json).unwrap();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].sequences.len(), 2);
    }

    #[test]
    fn test_chain_labels() {
        assert_eq!(chain_label(0), "A");
        assert_eq!(chain_label(25), "Z");
        assert_eq!(chain_label(26), "AA");
        assert_eq!(chain_label(27), "AB");
    }

    #[test]
    fn test_build_atom_array_protein() {
        let json = r#"[{
            "sequences": [
                {"proteinChain": {"sequence": "AG", "count": 1}}
            ],
            "name": "test"
        }]"#;
        let inputs = parse_inference_json_str(json).unwrap();
        let builder = SampleDictToFeatures::new(24);
        let array = builder.build_atom_array(&inputs[0]);
        assert!(array.len() > 0);
        assert_eq!(array.chain_id[0], "A");
        assert_eq!(array.mol_type[0], "protein");
    }
}
