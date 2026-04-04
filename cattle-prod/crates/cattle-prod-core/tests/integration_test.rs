use cattle_prod_core::config::*;
use cattle_prod_core::constants::*;
use cattle_prod_core::residue::*;
use cattle_prod_core::token::*;

// ---------------------------------------------------------------------------
// Constant table size tests
// ---------------------------------------------------------------------------

#[test]
fn std_residues_has_31_entries() {
    assert_eq!(STD_RESIDUES.len(), 31);
}

#[test]
fn std_residues_with_gap_has_32_entries() {
    assert_eq!(STD_RESIDUES_WITH_GAP.len(), 32);
}

#[test]
fn atom37_has_37_entries() {
    assert_eq!(ATOM37.len(), 37);
    assert_eq!(ATOM37_NUM, 37);
}

#[test]
fn atom37_vdw_has_37_entries() {
    assert_eq!(ATOM37_VDW.len(), 37);
}

#[test]
fn element_symbols_has_118_entries() {
    assert_eq!(ELEMENT_SYMBOLS.len(), 118);
}

#[test]
fn rdkit_vdws_has_118_entries() {
    assert_eq!(RDKIT_VDWS.len(), 118);
}

#[test]
fn chi_angles_mask_has_20_entries() {
    assert_eq!(CHI_ANGLES_MASK.len(), 20);
}

#[test]
fn protein_types_one_letter_has_20_entries() {
    assert_eq!(PROTEIN_TYPES_ONE_LETTER.len(), 20);
}

#[test]
fn crystallization_aids_has_16_entries() {
    assert_eq!(CRYSTALLIZATION_AIDS.len(), 16);
}

#[test]
fn crystallization_methods_has_5_entries() {
    assert_eq!(CRYSTALLIZATION_METHODS.len(), 5);
}

#[test]
fn atom37_order_has_37_entries() {
    assert_eq!(ATOM37_ORDER.len(), 37);
}

#[test]
fn num_std_residues_is_32() {
    assert_eq!(NUM_STD_RESIDUES, 32);
}

// ---------------------------------------------------------------------------
// Residue type lookup tests
// ---------------------------------------------------------------------------

#[test]
fn pro_std_residues_lookup_protein() {
    assert_eq!(PRO_STD_RESIDUES.get("ALA"), Some(&0));
    assert_eq!(PRO_STD_RESIDUES.get("GLY"), Some(&7));
    assert_eq!(PRO_STD_RESIDUES.get("VAL"), Some(&19));
    assert_eq!(PRO_STD_RESIDUES.get("UNK"), Some(&20));
    assert_eq!(PRO_STD_RESIDUES.len(), 21);
}

#[test]
fn rna_std_residues_lookup() {
    assert_eq!(RNA_STD_RESIDUES.get("A"), Some(&21));
    assert_eq!(RNA_STD_RESIDUES.get("G"), Some(&22));
    assert_eq!(RNA_STD_RESIDUES.get("C"), Some(&23));
    assert_eq!(RNA_STD_RESIDUES.get("U"), Some(&24));
    assert_eq!(RNA_STD_RESIDUES.get("N"), Some(&25));
    assert_eq!(RNA_STD_RESIDUES.len(), 5);
}

#[test]
fn dna_std_residues_lookup() {
    assert_eq!(DNA_STD_RESIDUES.get("DA"), Some(&26));
    assert_eq!(DNA_STD_RESIDUES.get("DG"), Some(&27));
    assert_eq!(DNA_STD_RESIDUES.get("DC"), Some(&28));
    assert_eq!(DNA_STD_RESIDUES.get("DT"), Some(&29));
    assert_eq!(DNA_STD_RESIDUES.get("DN"), Some(&30));
    assert_eq!(DNA_STD_RESIDUES.len(), 5);
}

#[test]
fn std_residues_unified_contains_all() {
    for (k, v) in PRO_STD_RESIDUES.iter() {
        assert_eq!(STD_RESIDUES.get(k), Some(v), "missing protein residue {k}");
    }
    for (k, v) in RNA_STD_RESIDUES.iter() {
        assert_eq!(STD_RESIDUES.get(k), Some(v), "missing RNA residue {k}");
    }
    for (k, v) in DNA_STD_RESIDUES.iter() {
        assert_eq!(STD_RESIDUES.get(k), Some(v), "missing DNA residue {k}");
    }
}

#[test]
fn std_residues_with_gap_includes_dash() {
    assert_eq!(STD_RESIDUES_WITH_GAP.get("-"), Some(&31));
}

#[test]
fn id_to_name_roundtrip() {
    for (&name, &id) in STD_RESIDUES_WITH_GAP.iter() {
        let recovered = STD_RESIDUES_WITH_GAP_ID_TO_NAME.get(&id).unwrap();
        assert_eq!(*recovered, name, "roundtrip failed for id={id}");
    }
}

#[test]
fn protein_one_to_three_letter_consistency() {
    for &letter in &PROTEIN_TYPES_ONE_LETTER {
        assert!(
            PROTEIN_COMMON_ONE_TO_THREE.contains_key(&letter),
            "missing one-letter code '{letter}' in PROTEIN_COMMON_ONE_TO_THREE"
        );
        assert!(
            PROT_STD_RESIDUES_ONE_TO_THREE.contains_key(&letter),
            "missing one-letter code '{letter}' in PROT_STD_RESIDUES_ONE_TO_THREE"
        );
    }
}

#[test]
fn mmcif_restype_roundtrip() {
    for (&one, &three) in MMCIF_RESTYPE_1TO3.iter() {
        let recovered = MMCIF_RESTYPE_3TO1.get(three);
        assert_eq!(recovered, Some(&one), "roundtrip failed for {one} -> {three}");
    }
}

// ---------------------------------------------------------------------------
// Atom37 order tests
// ---------------------------------------------------------------------------

#[test]
fn atom37_order_matches_array() {
    for (i, &name) in ATOM37.iter().enumerate() {
        assert_eq!(ATOM37_ORDER.get(name), Some(&i), "mismatch for {name}");
    }
}

#[test]
fn atom37_starts_with_backbone() {
    assert_eq!(ATOM37[0], "N");
    assert_eq!(ATOM37[1], "CA");
    assert_eq!(ATOM37[2], "C");
    assert_eq!(ATOM37[3], "CB");
    assert_eq!(ATOM37[4], "O");
}

// ---------------------------------------------------------------------------
// Element lookups
// ---------------------------------------------------------------------------

#[test]
fn elem_id_lookup() {
    assert!(elem_id("C").is_some());
    assert!(elem_id("N").is_some());
    assert!(elem_id("O").is_some());
    assert!(elem_id("H").is_some());
    assert!(elem_id("Fe").is_some());

    let c_id = elem_id("C").unwrap();
    let n_id = elem_id("N").unwrap();
    assert_ne!(c_id, n_id);
}

#[test]
fn elem_id_case_insensitive() {
    assert_eq!(elem_id("c"), elem_id("C"));
    assert_eq!(elem_id("fe"), elem_id("FE"));
    assert_eq!(elem_id("Fe"), elem_id("FE"));
}

#[test]
fn elem_id_unknown_returns_none() {
    assert!(elem_id("XX").is_none());
    assert!(elem_id("FAKE").is_none());
}

#[test]
fn elems_map_offset_by_num_std_residues() {
    let base = NUM_STD_RESIDUES as u32;
    let h_id = ELEMS.get("H").unwrap();
    assert_eq!(*h_id, base);

    let he_id = ELEMS.get("HE").unwrap();
    assert_eq!(*he_id, base + 1);
}

// ---------------------------------------------------------------------------
// RES_ATOMS_DICT tests
// ---------------------------------------------------------------------------

#[test]
fn res_atoms_dict_covers_standard_residues() {
    let expected_protein = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
    ];
    for res in expected_protein {
        assert!(
            RES_ATOMS_DICT.contains_key(res),
            "RES_ATOMS_DICT missing {res}"
        );
    }
}

#[test]
fn res_atoms_dict_covers_nucleotides() {
    let nuc = ["DA", "DC", "DG", "DT", "DN", "A", "C", "G", "U", "N"];
    for res in nuc {
        assert!(
            RES_ATOMS_DICT.contains_key(res),
            "RES_ATOMS_DICT missing {res}"
        );
    }
}

#[test]
fn gly_has_no_cb() {
    let gly = &RES_ATOMS_DICT["GLY"];
    assert!(!gly.contains_key("CB"), "GLY should not have CB");
    assert!(gly.contains_key("CA"), "GLY should have CA");
}

// ---------------------------------------------------------------------------
// DENSE_ATOM / ATOM14 tests
// ---------------------------------------------------------------------------

#[test]
fn atom14_gly_has_4_atoms() {
    assert_eq!(ATOM14["GLY"].len(), 4);
}

#[test]
fn dense_atom_includes_nucleotides() {
    assert!(DENSE_ATOM.contains_key("A"));
    assert!(DENSE_ATOM.contains_key("DA"));
    assert!(DENSE_ATOM.contains_key("G"));
    assert!(DENSE_ATOM.contains_key("DG"));
}

// ---------------------------------------------------------------------------
// VDW radii
// ---------------------------------------------------------------------------

#[test]
fn vdw_radii_positive() {
    for &r in &RDKIT_VDWS {
        assert!(r > 0.0, "VDW radius must be positive, got {r}");
    }
    for &r in &ATOM37_VDW {
        assert!(r > 0.0, "ATOM37 VDW radius must be positive, got {r}");
    }
}

// ---------------------------------------------------------------------------
// Token creation and annotation
// ---------------------------------------------------------------------------

#[test]
fn token_new_default() {
    let tok = Token::new(5);
    assert_eq!(tok.value, 5);
    assert!(tok.atom_indices.is_empty());
    assert!(tok.atom_names.is_empty());
    assert!(tok.annotations.is_empty());
}

#[test]
fn token_with_atoms() {
    let tok = Token::with_atoms(
        7,
        vec![0, 1, 2],
        vec!["N".into(), "CA".into(), "C".into()],
    );
    assert_eq!(tok.value, 7);
    assert_eq!(tok.atom_indices.len(), 3);
    assert_eq!(tok.atom_names[1], "CA");
}

#[test]
fn token_set_and_get_annotation() {
    let mut tok = Token::new(0);
    tok.set_annotation("res_name", AnnotValue::String("ALA".into()));
    tok.set_annotation("count", AnnotValue::Usize(42));
    tok.set_annotation("score", AnnotValue::F64(0.95));

    match tok.get_annotation("res_name").unwrap() {
        AnnotValue::String(s) => assert_eq!(s, "ALA"),
        other => panic!("expected String, got {:?}", other),
    }

    assert_eq!(tok.get_annotation("count").unwrap().as_usize(), Some(42));
    assert!((tok.get_annotation("score").unwrap().as_f64().unwrap() - 0.95).abs() < 1e-10);
    assert!(tok.get_annotation("nonexistent").is_none());
}

#[test]
fn token_centre_atom_index() {
    let mut tok = Token::new(0);
    assert!(tok.centre_atom_index().is_none());

    tok.set_annotation("centre_atom_index", AnnotValue::Usize(5));
    assert_eq!(tok.centre_atom_index(), Some(5));
}

// ---------------------------------------------------------------------------
// TokenArray operations
// ---------------------------------------------------------------------------

#[test]
fn token_array_basic_ops() {
    let tokens = vec![Token::new(1), Token::new(2), Token::new(3)];
    let ta = TokenArray::new(tokens);

    assert_eq!(ta.len(), 3);
    assert!(!ta.is_empty());
    assert_eq!(ta.get(0).unwrap().value, 1);
    assert_eq!(ta.get(2).unwrap().value, 3);
    assert!(ta.get(5).is_none());
}

#[test]
fn token_array_values() {
    let tokens = vec![Token::new(10), Token::new(20), Token::new(30)];
    let ta = TokenArray::new(tokens);
    assert_eq!(ta.values(), vec![10, 20, 30]);
}

#[test]
fn token_array_select() {
    let tokens = vec![Token::new(10), Token::new(20), Token::new(30), Token::new(40)];
    let ta = TokenArray::new(tokens);
    let selected = ta.select(&[1, 3]);

    assert_eq!(selected.len(), 2);
    assert_eq!(selected.values(), vec![20, 40]);
}

#[test]
fn token_array_set_centre_atom_indices() {
    let tokens = vec![Token::new(0), Token::new(1), Token::new(2)];
    let mut ta = TokenArray::new(tokens);
    ta.set_centre_atom_indices(&[10, 20, 30]);

    assert_eq!(ta.get(0).unwrap().centre_atom_index(), Some(10));
    assert_eq!(ta.get(1).unwrap().centre_atom_index(), Some(20));
    assert_eq!(ta.get(2).unwrap().centre_atom_index(), Some(30));
}

#[test]
fn token_array_annotation_vec() {
    let tokens = vec![Token::new(0), Token::new(1)];
    let mut ta = TokenArray::new(tokens);
    ta.set_annotation_vec(
        "chain",
        vec![AnnotValue::String("A".into()), AnnotValue::String("B".into())],
    );

    let annots = ta.get_annotation_vec("chain");
    assert_eq!(annots.len(), 2);
    assert!(annots[0].is_some());
    assert!(annots[1].is_some());
}

#[test]
fn token_array_iter() {
    let tokens = vec![Token::new(1), Token::new(2), Token::new(3)];
    let ta = TokenArray::new(tokens);
    let values: Vec<u32> = ta.iter().map(|t| t.value).collect();
    assert_eq!(values, vec![1, 2, 3]);
}

#[test]
fn token_array_into_iter() {
    let tokens = vec![Token::new(5), Token::new(6)];
    let ta = TokenArray::new(tokens);
    let mut sum = 0u32;
    for t in &ta {
        sum += t.value;
    }
    assert_eq!(sum, 11);
}

// ---------------------------------------------------------------------------
// Config serialization / deserialization roundtrip
// ---------------------------------------------------------------------------

#[test]
fn config_default_roundtrip_json() {
    let config = CattleProdConfig::default();
    let json = serde_json::to_string(&config).expect("serialize");
    let recovered: CattleProdConfig = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(recovered.model_name, config.model_name);
    assert_eq!(recovered.seeds, config.seeds);
    assert_eq!(recovered.dtype, config.dtype);
    assert_eq!(recovered.model.c_s, config.model.c_s);
    assert_eq!(recovered.model.c_z, config.model.c_z);
    assert_eq!(recovered.model.n_blocks, config.model.n_blocks);
}

#[test]
fn config_default_roundtrip_yaml() {
    let config = CattleProdConfig::default();
    let yaml = serde_yaml::to_string(&config).expect("serialize yaml");
    let recovered: CattleProdConfig = serde_yaml::from_str(&yaml).expect("deserialize yaml");

    assert_eq!(recovered.model.c_s, 384);
    assert_eq!(recovered.model.c_z, 128);
    assert_eq!(recovered.model.sigma_data, 16.0);
}

#[test]
fn model_config_defaults() {
    let mc = ModelConfig::default();
    assert_eq!(mc.c_s, 384);
    assert_eq!(mc.c_z, 128);
    assert_eq!(mc.c_s_inputs, 449);
    assert_eq!(mc.c_atom, 128);
    assert_eq!(mc.n_blocks, 48);
    assert_eq!(mc.max_atoms_per_token, 24);
    assert_eq!(mc.no_bins, 64);
    assert_eq!(mc.n_cycle, 10);
}

#[test]
fn pairformer_config_defaults() {
    let pc = PairformerConfig::default();
    assert_eq!(pc.n_blocks, 48);
    assert_eq!(pc.n_heads, 16);
    assert!((pc.dropout - 0.25).abs() < 1e-10);
}

#[test]
fn noise_scheduler_config_defaults() {
    let ns = NoiseSchedulerConfig::default();
    assert!((ns.s_max - 160.0).abs() < 1e-10);
    assert!((ns.s_min - 4e-4).abs() < 1e-10);
    assert!((ns.rho - 7.0).abs() < 1e-10);
    assert!((ns.sigma_data - 16.0).abs() < 1e-10);
}

#[test]
fn sample_diffusion_config_defaults() {
    let sd = SampleDiffusionConfig::default();
    assert_eq!(sd.n_step, 200);
    assert_eq!(sd.n_sample, 5);
    assert!((sd.gamma0 - 0.8).abs() < 1e-10);
}

#[test]
fn inference_config_defaults() {
    let ic = InferenceConfig::default();
    assert!(ic.sorted_by_ranking_score);
    assert!(!ic.need_atom_confidence);
    assert_eq!(ic.chunk_size, Some(256));
}

#[test]
fn data_config_all_false_by_default() {
    let dc = DataConfig::default();
    assert!(!dc.use_msa);
    assert!(!dc.use_template);
    assert!(!dc.use_rna_msa);
    assert!(!dc.esm_enable);
}

#[test]
fn config_from_json_partial() {
    let json = r#"{"model_name": "custom", "seeds": [1, 2, 3]}"#;
    let config: CattleProdConfig = serde_json::from_str(json).expect("parse partial");
    assert_eq!(config.model_name, "custom");
    assert_eq!(config.seeds, vec![1, 2, 3]);
}

// ---------------------------------------------------------------------------
// MSA sequence-to-ID mappings
// ---------------------------------------------------------------------------

#[test]
fn msa_protein_seq_to_id_standard_amino_acids() {
    assert_eq!(MSA_PROTEIN_SEQ_TO_ID.get(&'A'), Some(&0));
    assert_eq!(MSA_PROTEIN_SEQ_TO_ID.get(&'R'), Some(&1));
    assert_eq!(MSA_PROTEIN_SEQ_TO_ID.get(&'-'), Some(&31));
    assert_eq!(MSA_PROTEIN_SEQ_TO_ID.get(&'X'), Some(&20));
}

#[test]
fn msa_protein_seq_to_id_covers_alphabet() {
    let mapped_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-";
    for c in mapped_chars.chars() {
        assert!(
            MSA_PROTEIN_SEQ_TO_ID.contains_key(&c),
            "MSA_PROTEIN_SEQ_TO_ID missing '{c}'"
        );
    }
}

#[test]
fn msa_rna_seq_to_id_known_bases() {
    assert_eq!(MSA_RNA_SEQ_TO_ID.get(&'A'), Some(&21));
    assert_eq!(MSA_RNA_SEQ_TO_ID.get(&'G'), Some(&22));
    assert_eq!(MSA_RNA_SEQ_TO_ID.get(&'C'), Some(&23));
    assert_eq!(MSA_RNA_SEQ_TO_ID.get(&'U'), Some(&24));
    assert_eq!(MSA_RNA_SEQ_TO_ID.get(&'-'), Some(&31));
}

#[test]
fn msa_rna_unknown_maps_to_25() {
    assert_eq!(MSA_RNA_SEQ_TO_ID.get(&'X'), Some(&25));
    assert_eq!(MSA_RNA_SEQ_TO_ID.get(&'B'), Some(&25));
}

#[test]
fn msa_dna_seq_to_id_known_bases() {
    assert_eq!(MSA_DNA_SEQ_TO_ID.get(&'A'), Some(&26));
    assert_eq!(MSA_DNA_SEQ_TO_ID.get(&'G'), Some(&27));
    assert_eq!(MSA_DNA_SEQ_TO_ID.get(&'C'), Some(&28));
    assert_eq!(MSA_DNA_SEQ_TO_ID.get(&'T'), Some(&29));
    assert_eq!(MSA_DNA_SEQ_TO_ID.get(&'-'), Some(&31));
}

#[test]
fn msa_dna_unknown_maps_to_30() {
    assert_eq!(MSA_DNA_SEQ_TO_ID.get(&'X'), Some(&30));
}

// ---------------------------------------------------------------------------
// Entity poly type dict
// ---------------------------------------------------------------------------

#[test]
fn entity_poly_type_dict_categories() {
    assert!(ENTITY_POLY_TYPE_DICT.contains_key("protein"));
    assert!(ENTITY_POLY_TYPE_DICT.contains_key("nuc"));
    assert!(ENTITY_POLY_TYPE_DICT.contains_key("ligand"));

    let protein = &ENTITY_POLY_TYPE_DICT["protein"];
    assert!(protein.contains(&"polypeptide(L)"));

    let nuc = &ENTITY_POLY_TYPE_DICT["nuc"];
    assert!(nuc.contains(&"polyribonucleotide"));
    assert!(nuc.contains(&"polydeoxyribonucleotide"));
}

// ---------------------------------------------------------------------------
// Derived tables
// ---------------------------------------------------------------------------

#[test]
fn restype_pseudobeta_idx_has_correct_length() {
    let idx = make_restype_pseudobeta_idx();
    assert_eq!(idx.len(), NUM_STD_RESIDUES);
}

#[test]
fn restype_pseudobeta_gly_is_ca() {
    let idx = make_restype_pseudobeta_idx();
    let gly_type = *PRO_STD_RESIDUES.get("GLY").unwrap() as usize;
    let gly_atoms = &ATOM14["GLY"];
    let ca_pos = gly_atoms.iter().position(|&n| n == "CA").unwrap() as i32;
    assert_eq!(idx[gly_type], ca_pos, "GLY pseudobeta should be CA");
}

#[test]
fn restype_rigidgroup_dense_atom_idx_shape() {
    let arr = make_restype_rigidgroup_dense_atom_idx();
    assert_eq!(arr.shape(), &[NUM_STD_RESIDUES, 8, 3]);
}
