use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BondType {
    Covalent,
    Disulfide,
    Hydrogen,
    Metallic,
    Unknown,
}

impl BondType {
    pub fn from_cif_conn_type(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "disulf" => Self::Disulfide,
            "covale" | "covale_base" | "covale_phosph" | "covale_sugar" => Self::Covalent,
            "hydrog" => Self::Hydrogen,
            "metalc" => Self::Metallic,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Bond {
    pub atom_i: usize,
    pub atom_j: usize,
    pub bond_type: BondType,
}

#[derive(Debug, Clone)]
pub enum AnnotationData {
    I32(Vec<i32>),
    F32(Vec<f32>),
    Str(Vec<String>),
    U8(Vec<u8>),
    Usize(Vec<usize>),
}

impl AnnotationData {
    pub fn len(&self) -> usize {
        match self {
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Str(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::Usize(v) => v.len(),
        }
    }

    pub fn as_i32(&self) -> Option<&Vec<i32>> {
        match self { Self::I32(v) => Some(v), _ => None }
    }

    pub fn as_f32(&self) -> Option<&Vec<f32>> {
        match self { Self::F32(v) => Some(v), _ => None }
    }

    pub fn as_str_vec(&self) -> Option<&Vec<String>> {
        match self { Self::Str(v) => Some(v), _ => None }
    }

    pub fn as_u8(&self) -> Option<&Vec<u8>> {
        match self { Self::U8(v) => Some(v), _ => None }
    }

    pub fn as_usize(&self) -> Option<&Vec<usize>> {
        match self { Self::Usize(v) => Some(v), _ => None }
    }
}

/// Columnar atom storage mirroring biotite's AtomArray.
///
/// Each `Vec` has length equal to the number of atoms. Bonds and
/// extensible annotations are stored alongside the columnar data.
#[derive(Debug, Clone, Default)]
pub struct AtomArray {
    pub atom_name: Vec<String>,
    pub element: Vec<String>,
    pub res_name: Vec<String>,
    pub res_id: Vec<i32>,
    pub chain_id: Vec<String>,
    pub coord: Vec<[f32; 3]>,
    pub b_factor: Vec<f32>,
    pub occupancy: Vec<f32>,
    pub mol_type: Vec<String>,
    pub entity_id: Vec<i32>,
    pub centre_atom_mask: Vec<u8>,
    pub bonds: Vec<Bond>,
    annotations: HashMap<String, AnnotationData>,
}

impl AtomArray {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            atom_name: Vec::with_capacity(n),
            element: Vec::with_capacity(n),
            res_name: Vec::with_capacity(n),
            res_id: Vec::with_capacity(n),
            chain_id: Vec::with_capacity(n),
            coord: Vec::with_capacity(n),
            b_factor: Vec::with_capacity(n),
            occupancy: Vec::with_capacity(n),
            mol_type: Vec::with_capacity(n),
            entity_id: Vec::with_capacity(n),
            centre_atom_mask: Vec::with_capacity(n),
            bonds: Vec::new(),
            annotations: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.atom_name.len()
    }

    pub fn is_empty(&self) -> bool {
        self.atom_name.is_empty()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_atom(
        &mut self,
        atom_name: String,
        element: String,
        res_name: String,
        res_id: i32,
        chain_id: String,
        coord: [f32; 3],
        b_factor: f32,
        occupancy: f32,
        mol_type: String,
        entity_id: i32,
    ) {
        self.atom_name.push(atom_name);
        self.element.push(element);
        self.res_name.push(res_name);
        self.res_id.push(res_id);
        self.chain_id.push(chain_id);
        self.coord.push(coord);
        self.b_factor.push(b_factor);
        self.occupancy.push(occupancy);
        self.mol_type.push(mol_type);
        self.entity_id.push(entity_id);
        self.centre_atom_mask.push(0);
    }

    /// Return a new `AtomArray` containing only atoms where `predicate(index)` is true.
    pub fn filter<F: Fn(usize) -> bool>(&self, predicate: F) -> Self {
        let indices: Vec<usize> = (0..self.len()).filter(|&i| predicate(i)).collect();
        self.select_by_indices(&indices)
    }

    /// Return a new `AtomArray` containing atoms at the given indices (in order).
    pub fn select_by_indices(&self, indices: &[usize]) -> Self {
        let mut result = Self::with_capacity(indices.len());
        for &i in indices {
            result.atom_name.push(self.atom_name[i].clone());
            result.element.push(self.element[i].clone());
            result.res_name.push(self.res_name[i].clone());
            result.res_id.push(self.res_id[i]);
            result.chain_id.push(self.chain_id[i].clone());
            result.coord.push(self.coord[i]);
            result.b_factor.push(self.b_factor[i]);
            result.occupancy.push(self.occupancy[i]);
            result.mol_type.push(self.mol_type[i].clone());
            result.entity_id.push(self.entity_id[i]);
            result.centre_atom_mask.push(self.centre_atom_mask[i]);
        }

        let index_map: HashMap<usize, usize> = indices
            .iter()
            .enumerate()
            .map(|(new_idx, &old_idx)| (old_idx, new_idx))
            .collect();
        for bond in &self.bonds {
            if let (Some(&ni), Some(&nj)) =
                (index_map.get(&bond.atom_i), index_map.get(&bond.atom_j))
            {
                result.bonds.push(Bond {
                    atom_i: ni,
                    atom_j: nj,
                    bond_type: bond.bond_type,
                });
            }
        }

        for (key, data) in &self.annotations {
            let new_data = match data {
                AnnotationData::I32(v) => AnnotationData::I32(indices.iter().map(|&i| v[i]).collect()),
                AnnotationData::F32(v) => AnnotationData::F32(indices.iter().map(|&i| v[i]).collect()),
                AnnotationData::Str(v) => {
                    AnnotationData::Str(indices.iter().map(|&i| v[i].clone()).collect())
                }
                AnnotationData::U8(v) => AnnotationData::U8(indices.iter().map(|&i| v[i]).collect()),
                AnnotationData::Usize(v) => {
                    AnnotationData::Usize(indices.iter().map(|&i| v[i]).collect())
                }
            };
            result.annotations.insert(key.clone(), new_data);
        }
        result
    }

    /// Iterate over residue groups. Atoms are grouped by consecutive runs
    /// sharing the same `(chain_id, res_id, res_name)`.
    pub fn residue_iter(&self) -> ResidueIter<'_> {
        ResidueIter {
            array: self,
            pos: 0,
        }
    }

    /// Concatenate multiple `AtomArray`s, remapping bond indices.
    pub fn concat(arrays: &[&AtomArray]) -> Self {
        let total: usize = arrays.iter().map(|a| a.len()).sum();
        let mut result = Self::with_capacity(total);
        let mut offset = 0usize;

        for arr in arrays {
            result.atom_name.extend_from_slice(&arr.atom_name);
            result.element.extend_from_slice(&arr.element);
            result.res_name.extend_from_slice(&arr.res_name);
            result.res_id.extend_from_slice(&arr.res_id);
            result.chain_id.extend_from_slice(&arr.chain_id);
            result.coord.extend_from_slice(&arr.coord);
            result.b_factor.extend_from_slice(&arr.b_factor);
            result.occupancy.extend_from_slice(&arr.occupancy);
            result.mol_type.extend_from_slice(&arr.mol_type);
            result.entity_id.extend_from_slice(&arr.entity_id);
            result.centre_atom_mask.extend_from_slice(&arr.centre_atom_mask);

            for bond in &arr.bonds {
                result.bonds.push(Bond {
                    atom_i: bond.atom_i + offset,
                    atom_j: bond.atom_j + offset,
                    bond_type: bond.bond_type,
                });
            }
            offset += arr.len();
        }
        result
    }

    pub fn get_annotation(&self, key: &str) -> Option<&AnnotationData> {
        self.annotations.get(key)
    }

    pub fn set_annotation(&mut self, key: &str, data: AnnotationData) {
        self.annotations.insert(key.to_string(), data);
    }

    /// Look up an atom index by (chain_id, res_id, atom_name).
    pub fn find_atom(&self, chain: &str, res: i32, name: &str) -> Option<usize> {
        (0..self.len()).find(|&i| {
            self.chain_id[i] == chain && self.res_id[i] == res && self.atom_name[i] == name
        })
    }

    /// Return the mol_type enum string for the first atom of a residue.
    pub fn mol_type_of_residue(&self, start: usize) -> &str {
        &self.mol_type[start]
    }
}

/// Half-open range `[start, end)` identifying a residue group within an `AtomArray`.
#[derive(Debug, Clone, Copy)]
pub struct ResidueSlice {
    pub start: usize,
    pub end: usize,
}

impl ResidueSlice {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    pub fn indices(&self) -> std::ops::Range<usize> {
        self.start..self.end
    }
}

pub struct ResidueIter<'a> {
    array: &'a AtomArray,
    pos: usize,
}

impl<'a> Iterator for ResidueIter<'a> {
    type Item = ResidueSlice;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.array.len() {
            return None;
        }
        let start = self.pos;
        let chain = &self.array.chain_id[start];
        let res = self.array.res_id[start];
        let rn = &self.array.res_name[start];

        while self.pos < self.array.len()
            && self.array.chain_id[self.pos] == *chain
            && self.array.res_id[self.pos] == res
            && self.array.res_name[self.pos] == *rn
        {
            self.pos += 1;
        }
        Some(ResidueSlice {
            start,
            end: self.pos,
        })
    }
}
