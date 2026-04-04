use std::collections::HashMap;

use cattle_prod_data::dumper::{ConfidenceSummary, DataDumper};
use cattle_prod_data::featurizer::Featurizer;
use cattle_prod_data::metrics::{clash_score, gdt_ts, lddt, rmsd};
use cattle_prod_data::msa::{msa_profile, parse_a3m};
use cattle_prod_data::parser::CifParser;
use cattle_prod_data::template::{
    empty_template_features, TemplateAssemblyLine,
};
use cattle_prod_data::tokenizer::AtomArrayTokenizer;

// ---------------------------------------------------------------------------
// Minimal mmCIF content for a two-residue protein (ALA + GLY)
// ---------------------------------------------------------------------------

fn mini_cif() -> &'static str {
    r#"data_mini
#
loop_
_entity.id
_entity.type
1 polymer
#
loop_
_entity_poly.entity_id
_entity_poly.type
1 'polypeptide(L)'
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM 1  N  N   ALA A 1 1 A 1  0.000 0.000 0.000 1.00 10.0 1
ATOM 2  C  CA  ALA A 1 1 A 1  1.458 0.000 0.000 1.00 10.0 1
ATOM 3  C  C   ALA A 1 1 A 1  2.009 1.420 0.000 1.00 10.0 1
ATOM 4  O  O   ALA A 1 1 A 1  1.246 2.390 0.000 1.00 10.0 1
ATOM 5  C  CB  ALA A 1 1 A 1  1.986 -0.760 -1.208 1.00 10.0 1
ATOM 6  N  N   GLY A 1 2 A 2  3.320 1.480 0.000 1.00 10.0 1
ATOM 7  C  CA  GLY A 1 2 A 2  3.970 2.780 0.000 1.00 10.0 1
ATOM 8  C  C   GLY A 1 2 A 2  5.480 2.680 0.000 1.00 10.0 1
ATOM 9  O  O   GLY A 1 2 A 2  6.100 1.620 0.000 1.00 10.0 1
#
"#
}

// ===========================================================================
// CIF parsing -> tokenization -> featurization pipeline
// ===========================================================================

#[test]
fn parse_mini_cif_atom_count() {
    let aa = CifParser::parse_str(mini_cif()).unwrap();
    assert_eq!(aa.len(), 9, "mini CIF should have 9 atoms");
}

#[test]
fn parse_mini_cif_residue_groups() {
    let aa = CifParser::parse_str(mini_cif()).unwrap();
    let residues: Vec<_> = aa.residue_iter().collect();
    assert_eq!(residues.len(), 2, "should have 2 residues");
    assert_eq!(residues[0].len(), 5, "ALA has 5 atoms");
    assert_eq!(residues[1].len(), 4, "GLY has 4 atoms");
}

#[test]
fn parse_mini_cif_mol_type() {
    let aa = CifParser::parse_str(mini_cif()).unwrap();
    assert_eq!(aa.mol_type[0], "protein");
}

#[test]
fn tokenize_mini_cif() {
    let aa = CifParser::parse_str(mini_cif()).unwrap();
    let tokenizer = AtomArrayTokenizer::new(&aa);
    let ta = tokenizer.get_token_array();

    assert_eq!(ta.len(), 2, "protein residues yield one token each");
    assert!(
        ta.get(0).unwrap().centre_atom_index().is_some(),
        "centre atom should be set"
    );
}

#[test]
fn featurize_restype_onehot_shape() {
    let aa = CifParser::parse_str(mini_cif()).unwrap();
    let tokenizer = AtomArrayTokenizer::new(&aa);
    let ta = tokenizer.get_token_array();

    let feat = Featurizer::new(24);
    let onehot = feat.get_restype_onehot(&ta);

    assert_eq!(onehot.shape(), &[2, 32]);
    let row_sum: f32 = onehot.row(0).sum();
    assert!(
        (row_sum - 1.0).abs() < 1e-6,
        "one-hot row should sum to 1.0, got {row_sum}"
    );
}

#[test]
fn featurize_reference_features_shapes() {
    let aa = CifParser::parse_str(mini_cif()).unwrap();
    let tokenizer = AtomArrayTokenizer::new(&aa);
    let ta = tokenizer.get_token_array();

    let max_atoms = 24;
    let feat = Featurizer::new(max_atoms);
    let refs = feat.get_reference_features(&ta, &aa);

    assert_eq!(refs["ref_pos"].shape(), &[2, max_atoms, 3]);
    assert_eq!(refs["ref_mask"].shape(), &[2, max_atoms]);
    assert_eq!(refs["ref_element"].shape(), &[2, max_atoms, 128]);
    assert_eq!(refs["ref_space_uid"].shape(), &[2, max_atoms]);
}

#[test]
fn featurize_full_pipeline() {
    let aa = CifParser::parse_str(mini_cif()).unwrap();
    let tokenizer = AtomArrayTokenizer::new(&aa);
    let ta = tokenizer.get_token_array();

    let feat = Featurizer::new(24);
    let features = feat.featurize(&ta, &aa);

    assert!(features.contains_key("restype"));
    assert!(features.contains_key("token_bonds"));
    assert!(features.contains_key("ref_pos"));
    assert!(features.contains_key("atom_to_token_idx"));
    assert!(features.contains_key("residue_index"));
}

#[test]
fn featurize_atom_to_token_idx() {
    let aa = CifParser::parse_str(mini_cif()).unwrap();
    let tokenizer = AtomArrayTokenizer::new(&aa);
    let ta = tokenizer.get_token_array();

    let feat = Featurizer::new(24);
    let a2t = feat.get_atom_to_token_idx(&ta, aa.len());

    assert_eq!(a2t.len(), 9);
    for i in 0..5 {
        assert_eq!(a2t[i], 0, "ALA atoms should map to token 0");
    }
    for i in 5..9 {
        assert_eq!(a2t[i], 1, "GLY atoms should map to token 1");
    }
}

// ===========================================================================
// Template features – empty templates, padding, assembly
// ===========================================================================

#[test]
fn empty_template_features_shape() {
    let feats = empty_template_features(20);
    assert_eq!(feats.aatype.shape(), &[1, 20]);
    assert_eq!(feats.atom_positions.shape(), &[1, 20, 24, 3]);
    assert_eq!(feats.atom_mask.shape(), &[1, 20, 24]);
}

#[test]
fn empty_template_features_aatype_is_gap() {
    let feats = empty_template_features(5);
    for &v in feats.aatype.iter() {
        assert_eq!(v, 31, "empty template should use gap id (31)");
    }
}

#[test]
fn template_pad_to_max_and_reduce() {
    let feats = empty_template_features(10);
    assert_eq!(feats.num_templates(), 1);

    let padded = feats.pad_to_templates(4);
    assert_eq!(padded.aatype.shape(), &[4, 10]);
    assert_eq!(padded.atom_positions.shape(), &[4, 10, 24, 3]);

    let reduced = padded.reduce(2);
    assert_eq!(reduced.aatype.shape(), &[2, 10]);
}

#[test]
fn template_into_templates() {
    let raw = empty_template_features(6);
    let templates = raw.into_templates();
    assert_eq!(templates.num_templates(), 1);
    assert_eq!(templates.num_residues(), 6);
}

#[test]
fn template_assembly_line_basic() {
    let f1 = empty_template_features(4);
    let f2 = empty_template_features(3);
    let al = TemplateAssemblyLine::new(4);
    let indices: Vec<usize> = (0..7).collect();
    let templates = al.assemble(vec![f1, f2], &indices);

    assert_eq!(templates.num_templates(), 4);
    assert_eq!(templates.num_residues(), 7);
}

#[test]
fn template_assembly_line_empty_chains() {
    let al = TemplateAssemblyLine::new(4);
    let templates = al.assemble(vec![], &[]);

    assert_eq!(templates.num_templates(), 4);
    assert_eq!(templates.num_residues(), 0);
}

#[test]
fn template_assembly_line_subset_indices() {
    let f1 = empty_template_features(5);
    let al = TemplateAssemblyLine::new(2);
    let templates = al.assemble(vec![f1], &[0, 2, 4]);

    assert_eq!(templates.num_templates(), 2);
    assert_eq!(templates.num_residues(), 3);
}

// ===========================================================================
// MSA parsing -> profile computation
// ===========================================================================

fn sample_a3m() -> &'static str {
    ">query\nACDEF\n>hit1\nACDE-\n>hit2\nA-DEF\n"
}

#[test]
fn parse_a3m_basic_counts() {
    let msa = parse_a3m(sample_a3m());
    assert_eq!(msa.num_sequences(), 3);
    assert_eq!(msa.sequence_length(), 5);
}

#[test]
fn parse_a3m_aligned_sequences() {
    let msa = parse_a3m(sample_a3m());
    assert_eq!(&msa.sequences[0], "ACDEF");
    assert_eq!(&msa.sequences[1], "ACDE-");
    assert_eq!(&msa.sequences[2], "A-DEF");
}

#[test]
fn parse_a3m_insertions() {
    let content = ">query\nACaDE\n>hit1\nAC-DE\n";
    let msa = parse_a3m(content);
    assert_eq!(&msa.sequences[0], "ACDE");
    assert_eq!(msa.deletion_matrix[0], vec![0, 0, 1, 0]);
}

#[test]
fn msa_profile_shape() {
    let msa = parse_a3m(sample_a3m());
    let mut seq_to_id: HashMap<char, u8> = HashMap::new();
    for (i, ch) in "ACDEFGHIKLMNPQRSTVWY-".chars().enumerate() {
        seq_to_id.insert(ch, i as u8);
    }
    let num_classes = 21;
    let profile = msa_profile(&msa, &seq_to_id, num_classes);

    assert_eq!(profile.shape(), &[5, 21]);
}

#[test]
fn msa_profile_sums_to_one() {
    let msa = parse_a3m(sample_a3m());
    let mut seq_to_id: HashMap<char, u8> = HashMap::new();
    for (i, ch) in "ACDEFGHIKLMNPQRSTVWY-".chars().enumerate() {
        seq_to_id.insert(ch, i as u8);
    }
    let num_classes = 21;
    let profile = msa_profile(&msa, &seq_to_id, num_classes);

    for row_idx in 0..profile.shape()[0] {
        let row_sum: f32 = profile.row(row_idx).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-4,
            "profile row {row_idx} should sum to ~1.0, got {row_sum}"
        );
    }
}

#[test]
fn msa_profile_empty() {
    let msa = parse_a3m("");
    let seq_to_id: HashMap<char, u8> = HashMap::new();
    let profile = msa_profile(&msa, &seq_to_id, 21);
    assert_eq!(profile.shape(), &[0, 21]);
}

// ===========================================================================
// Metrics with known inputs
// ===========================================================================

#[test]
fn lddt_identical_structures_is_one() {
    let coords = vec![
        [0.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [7.6, 0.0, 0.0],
    ];
    let score = lddt(&coords, &coords, None, 1.0);
    assert!(
        (score - 1.0).abs() < 1e-9,
        "identical structures should give lDDT=1.0, got {score}"
    );
}

#[test]
fn lddt_with_mask() {
    let coords = vec![
        [0.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [100.0, 100.0, 100.0],
    ];
    let pred = vec![
        [0.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [50.0, 50.0, 50.0],
    ];
    let mask = vec![true, true, false];
    let score = lddt(&pred, &coords, Some(&mask), 1.0);
    assert!(
        (score - 1.0).abs() < 1e-9,
        "masked lDDT should be 1.0 for preserved pairs, got {score}"
    );
}

#[test]
fn lddt_empty_returns_zero() {
    let score = lddt(&[], &[], None, 1.0);
    assert!(score.abs() < 1e-9);
}

#[test]
fn rmsd_identical_is_zero() {
    let coords = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let r = rmsd(&coords, &coords, None);
    assert!(r.abs() < 1e-9, "RMSD of identical coords = 0, got {r}");
}

#[test]
fn rmsd_known_value() {
    let pred = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    let truth = vec![[1.0, 0.0, 0.0], [3.0, 1.0, 0.0]];
    let r = rmsd(&pred, &truth, None);
    let expected = ((1.0 + 1.0) / 2.0_f64).sqrt();
    assert!(
        (r - expected).abs() < 1e-6,
        "expected RMSD={expected}, got {r}"
    );
}

#[test]
fn rmsd_empty_returns_zero() {
    let r = rmsd(&[], &[], None);
    assert!(r.abs() < 1e-9);
}

#[test]
fn gdt_ts_identical_is_one() {
    let coords = vec![[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]];
    let score = gdt_ts(&coords, &coords);
    assert!(
        (score - 1.0).abs() < 1e-9,
        "GDT-TS of identical coords should be 1.0, got {score}"
    );
}

#[test]
fn gdt_ts_partial_within_thresholds() {
    let pred = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let truth = vec![[0.5, 0.0, 0.0], [3.0, 0.0, 0.0]];
    let score = gdt_ts(&pred, &truth);
    assert!(
        score > 0.0 && score < 1.0,
        "GDT-TS should be partial, got {score}"
    );
}

#[test]
fn clash_score_well_separated_is_zero() {
    let coords = vec![[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [40.0, 0.0, 0.0]];
    let radii = vec![1.7, 1.7, 1.7];
    let score = clash_score(&coords, &radii, 0.75);
    assert!(
        score.abs() < 1e-9,
        "no clashes expected for separated atoms, got {score}"
    );
}

#[test]
fn clash_score_overlapping_atoms() {
    let coords = vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
    let radii = vec![1.7, 1.7];
    let score = clash_score(&coords, &radii, 0.75);
    assert!(score > 0.0, "should detect clash, got {score}");
}

#[test]
fn clash_score_single_atom_is_zero() {
    let coords = vec![[0.0, 0.0, 0.0]];
    let radii = vec![1.7];
    let score = clash_score(&coords, &radii, 0.75);
    assert!(score.abs() < 1e-9);
}

// ===========================================================================
// DataDumper round-trip: write CIF -> re-read -> verify content
// ===========================================================================

#[test]
fn dumper_write_cif_and_reparse() {
    let dir = std::env::temp_dir().join("cattle_prod_integ_cif");
    let _ = std::fs::remove_dir_all(&dir);
    let dumper = DataDumper::new(dir.clone(), false, true);

    let coords = vec![[1.5, 2.5, 3.5], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let atom_names = vec!["N".to_string(), "CA".to_string(), "C".to_string()];
    let res_names = vec!["ALA".to_string(), "ALA".to_string(), "ALA".to_string()];
    let chain_ids = vec!["A".to_string(), "A".to_string(), "A".to_string()];
    let res_ids = vec![1, 1, 1];
    let elements = vec!["N".to_string(), "C".to_string(), "C".to_string()];

    let out_path = dir.join("round_trip.cif");
    dumper
        .write_cif(&coords, &atom_names, &res_names, &chain_ids, &res_ids, &elements, None, &out_path)
        .unwrap();

    let content = std::fs::read_to_string(&out_path).unwrap();
    assert!(content.contains("data_round_trip"));
    assert!(content.contains("_atom_site.Cartn_x"));

    let reparsed = CifParser::parse_str(&content).unwrap();
    assert_eq!(reparsed.len(), 3);
    assert_eq!(reparsed.atom_name[1], "CA");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dumper_confidence_json_roundtrip() {
    let dir = std::env::temp_dir().join("cattle_prod_integ_conf");
    let _ = std::fs::remove_dir_all(&dir);
    let dumper = DataDumper::new(dir.clone(), false, true);

    let summary = ConfidenceSummary {
        plddt: 85.0,
        ptm: 0.92,
        iptm: 0.88,
        ranking_score: 0.90,
        chain_ptm: vec![0.92],
        chain_plddt: vec![85.0],
    };

    let out_path = dir.join("confidence.json");
    dumper.write_confidence_json(&summary, &out_path).unwrap();

    let content = std::fs::read_to_string(&out_path).unwrap();
    let recovered: ConfidenceSummary = serde_json::from_str(&content).unwrap();

    assert!((recovered.plddt - 85.0).abs() < 1e-6);
    assert!((recovered.ptm - 0.92).abs() < 1e-6);
    assert!((recovered.ranking_score - 0.90).abs() < 1e-6);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dumper_full_dump_creates_tree() {
    let dir = std::env::temp_dir().join("cattle_prod_integ_dump_full");
    let _ = std::fs::remove_dir_all(&dir);
    let dumper = DataDumper::new(dir.clone(), true, true);

    let coords = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let names = vec!["CA".to_string(), "C".to_string()];
    let res = vec!["ALA".to_string(), "ALA".to_string()];
    let chains = vec!["A".to_string(), "A".to_string()];
    let ids = vec![1, 1];
    let elems = vec!["C".to_string(), "C".to_string()];
    let plddt = vec![80.0_f32, 75.0];
    let conf = ConfidenceSummary {
        plddt: 77.5,
        ptm: 0.9,
        iptm: 0.85,
        ranking_score: 0.87,
        chain_ptm: vec![0.9],
        chain_plddt: vec![77.5],
    };

    let sample_dir = dumper
        .dump("test_ds", "1ABC", 42, 0, 0, &coords, &names, &res, &chains, &ids, &elems, &conf, Some(&plddt))
        .unwrap();

    assert!(sample_dir.exists());
    assert!(sample_dir.join("rank_000_prediction.cif").exists());
    assert!(sample_dir.join("rank_000_confidence.json").exists());
    assert!(sample_dir.join("rank_000_plddt.json").exists());

    let cif_content = std::fs::read_to_string(sample_dir.join("rank_000_prediction.cif")).unwrap();
    let reparsed = CifParser::parse_str(&cif_content).unwrap();
    assert_eq!(reparsed.len(), 2);

    let _ = std::fs::remove_dir_all(&dir);
}

// ===========================================================================
// End-to-end: CIF -> tokenize -> featurize -> verify all shapes consistent
// ===========================================================================

#[test]
fn end_to_end_pipeline_consistency() {
    let aa = CifParser::parse_str(mini_cif()).unwrap();
    let n_atoms = aa.len();

    let tokenizer = AtomArrayTokenizer::new(&aa);
    let ta = tokenizer.get_token_array();
    let n_tokens = ta.len();

    let max_atoms = 24;
    let feat = Featurizer::new(max_atoms);

    let onehot = feat.get_restype_onehot(&ta);
    assert_eq!(onehot.shape()[0], n_tokens);

    let bonds = feat.get_token_bonds(&ta, &aa);
    assert_eq!(bonds.shape(), &[n_tokens, n_tokens]);

    let refs = feat.get_reference_features(&ta, &aa);
    assert_eq!(refs["ref_pos"].shape()[0], n_tokens);

    let a2t = feat.get_atom_to_token_idx(&ta, n_atoms);
    assert_eq!(a2t.len(), n_atoms);
    for &v in a2t.iter() {
        assert!(v >= 0, "all atoms should be assigned to a token");
    }

    let features = feat.featurize(&ta, &aa);
    assert!(features.len() >= 7);
}

// ===========================================================================
// CIF parsing edge cases
// ===========================================================================

#[test]
fn parse_empty_cif_fails() {
    let result = CifParser::parse_str("");
    assert!(result.is_err(), "empty CIF should fail");
}

#[test]
fn parse_cif_resolution() {
    let cif = "data_test\n_refine.ls_d_res_high 2.10\n";
    let res = CifParser::resolution(cif);
    assert!(res.is_some());
    assert!((res.unwrap() - 2.10).abs() < 1e-4);
}

#[test]
fn parse_cif_no_resolution() {
    let res = CifParser::resolution(mini_cif());
    assert!(res.is_none(), "mini CIF has no resolution");
}

#[test]
fn remove_water_from_parsed() {
    use cattle_prod_data::parser::remove_water;
    let aa = CifParser::parse_str(mini_cif()).unwrap();
    let filtered = remove_water(&aa);
    assert_eq!(filtered.len(), aa.len(), "mini CIF has no water");
}

#[test]
fn remove_hydrogen_from_parsed() {
    use cattle_prod_data::parser::remove_hydrogen;
    let aa = CifParser::parse_str(mini_cif()).unwrap();
    let filtered = remove_hydrogen(&aa);
    assert_eq!(filtered.len(), aa.len(), "mini CIF has no H atoms");
}
