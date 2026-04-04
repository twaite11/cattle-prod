use cattle_prod_core::constants::{elem_id, ELEMS};
use cattle_prod_core::residue::STD_RESIDUES;
use cattle_prod_core::token::{AnnotValue, Token, TokenArray};

use crate::atom_array::{AtomArray, ResidueSlice};

/// Centre-atom name used for each polymer type.
const PROTEIN_CENTRE_ATOM: &str = "CA";
const NUCLEIC_CENTRE_ATOM: &str = "C1'";

pub struct AtomArrayTokenizer<'a> {
    array: &'a AtomArray,
}

impl<'a> AtomArrayTokenizer<'a> {
    pub fn new(array: &'a AtomArray) -> Self {
        Self { array }
    }

    /// Build tokens from the atom array.
    ///
    /// For polymer residues whose `res_name` is in `STD_RESIDUES` (and whose
    /// mol_type is not ligand-like), a single token is emitted per residue with
    /// value equal to the standard-residue index.
    ///
    /// For all other residues (ligands, non-standard), one token is emitted per
    /// atom with its value derived from the element symbol via `ELEMS`.
    pub fn tokenize(&self) -> Vec<Token> {
        let mut tokens = Vec::new();

        for res in self.array.residue_iter() {
            let res_name = &self.array.res_name[res.start];
            let mol = &self.array.mol_type[res.start];
            let is_ligand = mol == "ligand" || mol == "non-polymer" || mol == "branched";

            if let Some(&std_id) = STD_RESIDUES.get(res_name.as_str()) {
                if !is_ligand {
                    tokens.push(self.residue_token(std_id as u32, &res));
                    continue;
                }
            }

            // Non-standard / ligand: one token per atom
            for i in res.start..res.end {
                let elem_upper = self.array.element[i].to_uppercase();
                let val = elem_id(&elem_upper).unwrap_or_else(|| {
                    ELEMS.get("C").copied().unwrap_or(38) // fall back to carbon
                });
                let mut tok = Token::with_atoms(
                    val,
                    vec![i],
                    vec![self.array.atom_name[i].clone()],
                );
                tok.set_annotation("res_name", AnnotValue::String(res_name.clone()));
                tok.set_annotation("chain_id", AnnotValue::String(
                    self.array.chain_id[i].clone(),
                ));
                tok.set_annotation("res_id", AnnotValue::I64(self.array.res_id[i] as i64));
                tok.set_annotation("is_ligand", AnnotValue::Usize(1));
                tokens.push(tok);
            }
        }
        tokens
    }

    /// Tokenize and wrap into a `TokenArray`, setting centre-atom indices.
    pub fn get_token_array(&self) -> TokenArray {
        let tokens = self.tokenize();
        let mut ta = TokenArray::new(tokens);
        self.set_centre_atom_indices(&mut ta);
        ta
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn residue_token(&self, value: u32, res: &ResidueSlice) -> Token {
        let indices: Vec<usize> = (res.start..res.end).collect();
        let names: Vec<String> = indices
            .iter()
            .map(|&i| self.array.atom_name[i].clone())
            .collect();
        let res_name = self.array.res_name[res.start].clone();
        let chain_id = self.array.chain_id[res.start].clone();
        let res_id = self.array.res_id[res.start];

        let mut tok = Token::with_atoms(value, indices, names);
        tok.set_annotation("res_name", AnnotValue::String(res_name));
        tok.set_annotation("chain_id", AnnotValue::String(chain_id));
        tok.set_annotation("res_id", AnnotValue::I64(res_id as i64));
        tok.set_annotation("is_ligand", AnnotValue::Usize(0));
        tok
    }

    /// For every token find the centre atom and store its global index.
    fn set_centre_atom_indices(&self, ta: &mut TokenArray) {
        let mut centres = Vec::with_capacity(ta.len());

        for token in ta.iter() {
            let centre = if token.atom_indices.len() == 1 {
                // ligand atom – the atom itself is the centre
                token.atom_indices[0]
            } else {
                self.find_centre_atom(token)
            };
            centres.push(centre);
        }

        ta.set_centre_atom_indices(&centres);
    }

    fn find_centre_atom(&self, token: &Token) -> usize {
        let mol = self.array.mol_type[token.atom_indices[0]].as_str();
        let target = match mol {
            "protein" => PROTEIN_CENTRE_ATOM,
            "rna" | "dna" => NUCLEIC_CENTRE_ATOM,
            _ => PROTEIN_CENTRE_ATOM,
        };

        for &idx in &token.atom_indices {
            if self.array.atom_name[idx] == target {
                return idx;
            }
        }
        // Fallback: first atom in the token
        token.atom_indices[0]
    }
}
