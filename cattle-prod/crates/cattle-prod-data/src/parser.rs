use std::collections::HashMap;
use std::io::{BufReader, Read};

use anyhow::{Context, Result};

use crate::atom_array::{AtomArray, Bond, BondType};

// ---------------------------------------------------------------------------
// Low-level CIF structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct CifLoop {
    tags: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl CifLoop {
    fn col_index(&self, tag: &str) -> Option<usize> {
        self.tags.iter().position(|t| t == tag)
    }

    #[allow(dead_code)]
    fn get_col(&self, tag: &str) -> Option<Vec<&str>> {
        let idx = self.col_index(tag)?;
        Some(self.rows.iter().map(|r| r[idx].as_str()).collect())
    }
}

#[derive(Debug, Default)]
pub(crate) struct CifBlock {
    data_items: HashMap<String, String>,
    loops: Vec<CifLoop>,
}

impl CifBlock {
    fn find_loop(&self, category_prefix: &str) -> Option<&CifLoop> {
        self.loops
            .iter()
            .find(|lp| lp.tags.first().map_or(false, |t| t.starts_with(category_prefix)))
    }
}

// ---------------------------------------------------------------------------
// CIF tokeniser – splits a single line respecting quotes
// ---------------------------------------------------------------------------

fn tokenize_cif_line(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        let b = bytes[i];
        if b == b' ' || b == b'\t' {
            i += 1;
            continue;
        }
        if b == b'#' {
            break;
        }
        if b == b'\'' || b == b'"' {
            let quote = b;
            i += 1;
            let start = i;
            while i < len && bytes[i] != quote {
                i += 1;
            }
            tokens.push(String::from_utf8_lossy(&bytes[start..i]).into_owned());
            if i < len {
                i += 1; // skip closing quote
            }
        } else {
            let start = i;
            while i < len && bytes[i] != b' ' && bytes[i] != b'\t' {
                i += 1;
            }
            tokens.push(String::from_utf8_lossy(&bytes[start..i]).into_owned());
        }
    }
    tokens
}

// ---------------------------------------------------------------------------
// CIF block parser
// ---------------------------------------------------------------------------

#[derive(PartialEq)]
enum CifState {
    Idle,
    LoopHeader,
    LoopData,
    SemicolonString,
}

fn parse_cif_block(content: &str) -> CifBlock {
    let mut block = CifBlock::default();
    let mut state = CifState::Idle;
    let mut current_loop = CifLoop::default();
    let mut pending: Vec<String> = Vec::new();

    let mut semi_buf = String::new();
    let mut semi_is_data_item = false;
    let mut semi_key = String::new();

    let flush = |pending: &mut Vec<String>, lp: &mut CifLoop| {
        let ncols = lp.tags.len();
        if ncols == 0 {
            return;
        }
        while pending.len() >= ncols {
            let row: Vec<String> = pending.drain(..ncols).collect();
            lp.rows.push(row);
        }
    };

    for line in content.lines() {
        // --- semicolon strings ---
        if state == CifState::SemicolonString {
            if line.starts_with(';') {
                if semi_is_data_item {
                    block.data_items.insert(semi_key.clone(), semi_buf.clone());
                } else {
                    pending.push(semi_buf.clone());
                    flush(&mut pending, &mut current_loop);
                }
                semi_buf.clear();
                state = if current_loop.tags.is_empty() {
                    CifState::Idle
                } else {
                    CifState::LoopData
                };
                continue;
            }
            if !semi_buf.is_empty() {
                semi_buf.push('\n');
            }
            semi_buf.push_str(line);
            continue;
        }

        let trimmed = line.trim();

        // detect semicolon string start (must be column-0 semicolon)
        if line.starts_with(';') {
            state = CifState::SemicolonString;
            semi_buf = line[1..].to_string();
            semi_is_data_item = current_loop.tags.is_empty();
            continue;
        }

        if trimmed.is_empty() || trimmed.starts_with('#') {
            if state == CifState::LoopData {
                flush(&mut pending, &mut current_loop);
                if !current_loop.tags.is_empty() {
                    block.loops.push(std::mem::take(&mut current_loop));
                }
                pending.clear();
                state = CifState::Idle;
            }
            continue;
        }

        if trimmed.starts_with("data_") {
            if state == CifState::LoopData {
                flush(&mut pending, &mut current_loop);
                if !current_loop.tags.is_empty() {
                    block.loops.push(std::mem::take(&mut current_loop));
                }
                pending.clear();
            }
            state = CifState::Idle;
            continue;
        }

        if trimmed == "loop_" {
            if state == CifState::LoopData {
                flush(&mut pending, &mut current_loop);
                if !current_loop.tags.is_empty() {
                    block.loops.push(std::mem::take(&mut current_loop));
                }
                pending.clear();
            }
            current_loop = CifLoop::default();
            state = CifState::LoopHeader;
            continue;
        }

        match state {
            CifState::LoopHeader => {
                if trimmed.starts_with('_') {
                    current_loop.tags.push(trimmed.to_string());
                } else {
                    state = CifState::LoopData;
                    let vals = tokenize_cif_line(trimmed);
                    pending.extend(vals);
                    flush(&mut pending, &mut current_loop);
                }
            }
            CifState::LoopData => {
                if trimmed.starts_with('_') || trimmed.starts_with("loop_") {
                    flush(&mut pending, &mut current_loop);
                    if !current_loop.tags.is_empty() {
                        block.loops.push(std::mem::take(&mut current_loop));
                    }
                    pending.clear();

                    if trimmed == "loop_" {
                        current_loop = CifLoop::default();
                        state = CifState::LoopHeader;
                    } else {
                        state = CifState::Idle;
                        let vals = tokenize_cif_line(trimmed);
                        if vals.len() >= 2 {
                            block.data_items.insert(vals[0].clone(), vals[1].clone());
                        } else if vals.len() == 1 {
                            semi_key = vals[0].clone();
                            semi_is_data_item = true;
                        }
                    }
                } else {
                    let vals = tokenize_cif_line(trimmed);
                    pending.extend(vals);
                    flush(&mut pending, &mut current_loop);
                }
            }
            CifState::Idle => {
                if trimmed.starts_with('_') {
                    let vals = tokenize_cif_line(trimmed);
                    if vals.len() >= 2 {
                        block.data_items.insert(vals[0].clone(), vals[1].clone());
                    } else if vals.len() == 1 {
                        semi_key = vals[0].clone();
                        semi_is_data_item = true;
                    }
                }
            }
            CifState::SemicolonString => unreachable!(),
        }
    }

    if state == CifState::LoopData {
        flush(&mut pending, &mut current_loop);
    }
    if !current_loop.tags.is_empty() {
        block.loops.push(current_loop);
    }
    block
}

// ---------------------------------------------------------------------------
// Entity type resolution
// ---------------------------------------------------------------------------

fn resolve_entity_mol_types(block: &CifBlock) -> HashMap<String, String> {
    let mut out: HashMap<String, String> = HashMap::new();

    if let Some(lp) = block.find_loop("_entity.") {
        let id_col = lp.col_index("_entity.id");
        let type_col = lp.col_index("_entity.type");
        if let (Some(ic), Some(tc)) = (id_col, type_col) {
            for row in &lp.rows {
                let eid = &row[ic];
                let etype = row[tc].to_lowercase();
                let mol = match etype.as_str() {
                    "polymer" => "polymer".to_string(),
                    "non-polymer" => "ligand".to_string(),
                    "water" => "water".to_string(),
                    "branched" => "ligand".to_string(),
                    other => other.to_string(),
                };
                out.insert(eid.clone(), mol);
            }
        }
    }

    if let Some(lp) = block.find_loop("_entity_poly.") {
        let id_col = lp.col_index("_entity_poly.entity_id");
        let type_col = lp.col_index("_entity_poly.type");
        if let (Some(ic), Some(tc)) = (id_col, type_col) {
            for row in &lp.rows {
                let eid = &row[ic];
                let ptype = row[tc].to_lowercase();
                let mol = if ptype.contains("polypeptide") {
                    "protein"
                } else if ptype.contains("polyribonucleotide") && !ptype.contains("deoxy") {
                    "rna"
                } else if ptype.contains("polydeoxyribonucleotide") {
                    "dna"
                } else {
                    "ligand"
                };
                out.insert(eid.clone(), mol.to_string());
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Build AtomArray from _atom_site records
// ---------------------------------------------------------------------------

fn build_atom_array(block: &CifBlock) -> Result<AtomArray> {
    let lp = block
        .find_loop("_atom_site.")
        .context("mmCIF file missing _atom_site loop")?;

    let tag = |name: &str| lp.col_index(name);

    let i_group = tag("_atom_site.group_PDB");
    let i_atom_id = tag("_atom_site.label_atom_id");
    let i_comp = tag("_atom_site.label_comp_id");
    let i_asym_auth = tag("_atom_site.auth_asym_id");
    let i_asym_label = tag("_atom_site.label_asym_id");
    let i_entity = tag("_atom_site.label_entity_id");
    let i_seq_auth = tag("_atom_site.auth_seq_id");
    let i_seq_label = tag("_atom_site.label_seq_id");
    let i_x = tag("_atom_site.Cartn_x");
    let i_y = tag("_atom_site.Cartn_y");
    let i_z = tag("_atom_site.Cartn_z");
    let i_elem = tag("_atom_site.type_symbol");
    let i_bfac = tag("_atom_site.B_iso_or_equiv");
    let i_occ = tag("_atom_site.occupancy");
    let i_model = tag("_atom_site.pdbx_PDB_model_num");

    let entity_types = resolve_entity_mol_types(block);

    let mut array = AtomArray::with_capacity(lp.rows.len());
    let mut first_model: Option<&str> = None;

    for row in &lp.rows {
        if let Some(mi) = i_model {
            let model_str = row[mi].as_str();
            match &first_model {
                None => first_model = Some(model_str),
                Some(fm) => {
                    if model_str != *fm {
                        continue;
                    }
                }
            }
        }

        // skip alt-loc duplicates: keep first occurrence only (handled by occupancy later)
        let _group = i_group.map(|i| row[i].as_str()).unwrap_or("ATOM");

        let atom_name = i_atom_id.map(|i| row[i].clone()).unwrap_or_default();
        let res_name = i_comp.map(|i| row[i].clone()).unwrap_or_default();

        let chain_id = i_asym_auth
            .map(|i| row[i].clone())
            .or_else(|| i_asym_label.map(|i| row[i].clone()))
            .unwrap_or_default();

        let entity_id_str = i_entity.map(|i| row[i].as_str()).unwrap_or("0");
        let entity_id: i32 = entity_id_str.parse().unwrap_or(0);

        let res_id_str = i_seq_auth
            .map(|i| row[i].as_str())
            .or_else(|| i_seq_label.map(|i| row[i].as_str()))
            .unwrap_or("0");
        let res_id: i32 = res_id_str.parse().unwrap_or(0);

        let parse_f = |idx: Option<usize>| -> f32 {
            idx.and_then(|i| row[i].parse::<f32>().ok()).unwrap_or(0.0)
        };
        let x = parse_f(i_x);
        let y = parse_f(i_y);
        let z = parse_f(i_z);
        let bfac = parse_f(i_bfac);
        let occ = parse_f(i_occ);

        let element = i_elem
            .map(|i| row[i].to_uppercase())
            .unwrap_or_default();

        let mol_type = entity_types
            .get(entity_id_str)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        array.push_atom(
            atom_name, element, res_name, res_id, chain_id, [x, y, z], bfac, occ, mol_type,
            entity_id,
        );
    }

    extract_bonds(block, &mut array);
    Ok(array)
}

// ---------------------------------------------------------------------------
// Bond extraction from _struct_conn
// ---------------------------------------------------------------------------

fn extract_bonds(block: &CifBlock, array: &mut AtomArray) {
    let lp = match block.find_loop("_struct_conn.") {
        Some(l) => l,
        None => return,
    };

    let tag = |name: &str| lp.col_index(name);
    let i_type = tag("_struct_conn.conn_type_id");
    let i_c1 = tag("_struct_conn.ptnr1_auth_asym_id")
        .or_else(|| tag("_struct_conn.ptnr1_label_asym_id"));
    let i_r1 = tag("_struct_conn.ptnr1_auth_seq_id")
        .or_else(|| tag("_struct_conn.ptnr1_label_seq_id"));
    let i_a1 = tag("_struct_conn.ptnr1_label_atom_id");
    let i_c2 = tag("_struct_conn.ptnr2_auth_asym_id")
        .or_else(|| tag("_struct_conn.ptnr2_label_asym_id"));
    let i_r2 = tag("_struct_conn.ptnr2_auth_seq_id")
        .or_else(|| tag("_struct_conn.ptnr2_label_seq_id"));
    let i_a2 = tag("_struct_conn.ptnr2_label_atom_id");

    for row in &lp.rows {
        let bt = i_type
            .map(|i| BondType::from_cif_conn_type(&row[i]))
            .unwrap_or(BondType::Unknown);

        let get_s = |idx: Option<usize>| idx.map(|i| row[i].as_str()).unwrap_or("");
        let chain1 = get_s(i_c1);
        let res1: i32 = i_r1
            .and_then(|i| row[i].parse().ok())
            .unwrap_or(0);
        let atom1 = get_s(i_a1);
        let chain2 = get_s(i_c2);
        let res2: i32 = i_r2
            .and_then(|i| row[i].parse().ok())
            .unwrap_or(0);
        let atom2 = get_s(i_a2);

        if let (Some(ai), Some(aj)) = (
            array.find_atom(chain1, res1, atom1),
            array.find_atom(chain2, res2, atom2),
        ) {
            array.bonds.push(Bond {
                atom_i: ai,
                atom_j: aj,
                bond_type: bt,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Assembly expansion
// ---------------------------------------------------------------------------

struct TransformOp {
    id: String,
    matrix: [[f64; 3]; 3],
    vector: [f64; 3],
}

fn parse_assembly_ops(block: &CifBlock) -> Vec<TransformOp> {
    let lp = match block.find_loop("_pdbx_struct_oper_list.") {
        Some(l) => l,
        None => return Vec::new(),
    };

    let tag = |n: &str| lp.col_index(n);
    let i_id = match tag("_pdbx_struct_oper_list.id") {
        Some(i) => i,
        None => return Vec::new(),
    };

    let mat_tags: [[&str; 3]; 3] = [
        [
            "_pdbx_struct_oper_list.matrix[1][1]",
            "_pdbx_struct_oper_list.matrix[1][2]",
            "_pdbx_struct_oper_list.matrix[1][3]",
        ],
        [
            "_pdbx_struct_oper_list.matrix[2][1]",
            "_pdbx_struct_oper_list.matrix[2][2]",
            "_pdbx_struct_oper_list.matrix[2][3]",
        ],
        [
            "_pdbx_struct_oper_list.matrix[3][1]",
            "_pdbx_struct_oper_list.matrix[3][2]",
            "_pdbx_struct_oper_list.matrix[3][3]",
        ],
    ];
    let vec_tags = [
        "_pdbx_struct_oper_list.vector[1]",
        "_pdbx_struct_oper_list.vector[2]",
        "_pdbx_struct_oper_list.vector[3]",
    ];

    let mi: [[Option<usize>; 3]; 3] = mat_tags.map(|row| row.map(|t| tag(t)));
    let vi: [Option<usize>; 3] = vec_tags.map(|t| tag(t));

    let mut ops = Vec::new();
    for row in &lp.rows {
        let pf = |idx: Option<usize>| -> f64 {
            idx.and_then(|i| row[i].parse().ok()).unwrap_or(0.0)
        };
        let matrix = [
            [pf(mi[0][0]), pf(mi[0][1]), pf(mi[0][2])],
            [pf(mi[1][0]), pf(mi[1][1]), pf(mi[1][2])],
            [pf(mi[2][0]), pf(mi[2][1]), pf(mi[2][2])],
        ];
        let vector = [pf(vi[0]), pf(vi[1]), pf(vi[2])];
        ops.push(TransformOp {
            id: row[i_id].clone(),
            matrix,
            vector,
        });
    }
    ops
}

fn is_identity(op: &TransformOp) -> bool {
    let eps = 1e-6;
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (op.matrix[i][j] - expected).abs() > eps {
                return false;
            }
        }
        if op.vector[i].abs() > eps {
            return false;
        }
    }
    true
}

fn apply_transform(coord: [f32; 3], op: &TransformOp) -> [f32; 3] {
    let c = [coord[0] as f64, coord[1] as f64, coord[2] as f64];
    let m = &op.matrix;
    let v = &op.vector;
    [
        (m[0][0] * c[0] + m[0][1] * c[1] + m[0][2] * c[2] + v[0]) as f32,
        (m[1][0] * c[0] + m[1][1] * c[1] + m[1][2] * c[2] + v[1]) as f32,
        (m[2][0] * c[0] + m[2][1] * c[1] + m[2][2] * c[2] + v[2]) as f32,
    ]
}

/// Expand biological assembly: duplicate chains and apply symmetry transforms.
pub(crate) fn expand_assembly(block: &CifBlock, array: &AtomArray, assembly_id: &str) -> AtomArray {
    let ops = parse_assembly_ops(block);
    if ops.is_empty() {
        return array.clone();
    }

    let ops_map: HashMap<&str, &TransformOp> =
        ops.iter().map(|o| (o.id.as_str(), o)).collect();

    let gen_loop = match block.find_loop("_pdbx_struct_assembly_gen.") {
        Some(l) => l,
        None => return array.clone(),
    };
    let i_aid = gen_loop.col_index("_pdbx_struct_assembly_gen.assembly_id");
    let i_oper = gen_loop.col_index("_pdbx_struct_assembly_gen.oper_expression");
    let i_asym = gen_loop.col_index("_pdbx_struct_assembly_gen.asym_id_list");

    let (i_aid, i_oper, i_asym) = match (i_aid, i_oper, i_asym) {
        (Some(a), Some(o), Some(s)) => (a, o, s),
        _ => return array.clone(),
    };

    let mut result = AtomArray::new();

    for row in &gen_loop.rows {
        if row[i_aid] != assembly_id {
            continue;
        }
        let oper_ids: Vec<&str> = row[i_oper]
            .split(',')
            .map(|s| s.trim().trim_matches(|c| c == '(' || c == ')'))
            .collect();
        let chain_ids: Vec<&str> = row[i_asym].split(',').map(|s| s.trim()).collect();

        for oper_id in &oper_ids {
            let op = match ops_map.get(oper_id) {
                Some(o) => o,
                None => continue,
            };

            for &cid in &chain_ids {
                for i in 0..array.len() {
                    if array.chain_id[i] != cid {
                        continue;
                    }
                    let new_coord = if is_identity(op) {
                        array.coord[i]
                    } else {
                        apply_transform(array.coord[i], op)
                    };
                    let suffix = if is_identity(op) {
                        String::new()
                    } else {
                        format!("_{}", oper_id)
                    };
                    result.push_atom(
                        array.atom_name[i].clone(),
                        array.element[i].clone(),
                        array.res_name[i].clone(),
                        array.res_id[i],
                        format!("{}{}", cid, suffix),
                        new_coord,
                        array.b_factor[i],
                        array.occupancy[i],
                        array.mol_type[i].clone(),
                        array.entity_id[i],
                    );
                }
            }
        }
    }

    if result.is_empty() {
        array.clone()
    } else {
        result
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct CifParser;

impl CifParser {
    /// Parse an mmCIF file at `path` and return an `AtomArray` (first model only).
    pub fn parse_file(path: &str) -> Result<AtomArray> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read CIF file: {}", path))?;
        Self::parse_str(&content)
    }

    /// Parse mmCIF content from a reader.
    pub fn parse_reader<R: std::io::Read>(reader: R) -> Result<AtomArray> {
        let mut buf = String::new();
        let mut br = BufReader::new(reader);
        br.read_to_string(&mut buf)?;
        Self::parse_str(&buf)
    }

    /// Parse mmCIF content from a string slice.
    pub fn parse_str(content: &str) -> Result<AtomArray> {
        let block = parse_cif_block(content);
        build_atom_array(&block)
    }

    /// Parse and apply biological assembly expansion.
    pub fn parse_with_assembly(content: &str, assembly_id: &str) -> Result<AtomArray> {
        let block = parse_cif_block(content);
        let array = build_atom_array(&block)?;
        Ok(expand_assembly(&block, &array, assembly_id))
    }

    /// Return resolution from `_refine.ls_d_res_high`, if present.
    pub fn resolution(content: &str) -> Option<f32> {
        let block = parse_cif_block(content);
        block
            .data_items
            .get("_refine.ls_d_res_high")
            .and_then(|v| v.parse::<f32>().ok())
            .or_else(|| {
                block
                    .find_loop("_refine.")
                    .and_then(|lp| {
                        let ci = lp.col_index("_refine.ls_d_res_high")?;
                        lp.rows.first().and_then(|r| r[ci].parse().ok())
                    })
            })
    }
}

// ---------------------------------------------------------------------------
// Convenience filters
// ---------------------------------------------------------------------------

pub fn remove_water(array: &AtomArray) -> AtomArray {
    array.filter(|i| {
        let rn = array.res_name[i].as_str();
        rn != "HOH" && rn != "WAT" && rn != "DOD"
    })
}

pub fn remove_hydrogen(array: &AtomArray) -> AtomArray {
    array.filter(|i| {
        let e = array.element[i].as_str();
        e != "H" && e != "D"
    })
}

/// Return true if the structure's resolution is within `max_resolution` Å.
pub fn filter_by_resolution(content: &str, max_resolution: f32) -> bool {
    CifParser::resolution(content)
        .map(|r| r <= max_resolution)
        .unwrap_or(true)
}
