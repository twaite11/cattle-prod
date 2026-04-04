use criterion::{black_box, criterion_group, criterion_main, Criterion};

use cattle_prod_data::msa::{msa_profile, parse_a3m};
use cattle_prod_data::parser::CifParser;
use cattle_prod_data::tokenizer::AtomArrayTokenizer;

// ---------------------------------------------------------------------------
// Sample data generators
// ---------------------------------------------------------------------------

fn sample_cif_content() -> String {
    let mut s = String::from("data_SAMPLE\n#\nloop_\n");
    s.push_str("_atom_site.group_PDB\n");
    s.push_str("_atom_site.label_atom_id\n");
    s.push_str("_atom_site.label_comp_id\n");
    s.push_str("_atom_site.auth_asym_id\n");
    s.push_str("_atom_site.label_asym_id\n");
    s.push_str("_atom_site.label_entity_id\n");
    s.push_str("_atom_site.auth_seq_id\n");
    s.push_str("_atom_site.label_seq_id\n");
    s.push_str("_atom_site.Cartn_x\n");
    s.push_str("_atom_site.Cartn_y\n");
    s.push_str("_atom_site.Cartn_z\n");
    s.push_str("_atom_site.type_symbol\n");
    s.push_str("_atom_site.B_iso_or_equiv\n");
    s.push_str("_atom_site.occupancy\n");
    s.push_str("_atom_site.pdbx_PDB_model_num\n");

    let backbone = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")];
    let residues = ["ALA", "GLY", "VAL", "LEU", "ILE", "PRO", "PHE", "TRP", "MET", "SER"];

    for (res_idx, res_name) in residues.iter().enumerate() {
        let seq_id = res_idx + 1;
        let atoms = if *res_name == "GLY" { &backbone[..4] } else { &backbone[..] };
        for (atom_idx, (atom_name, elem)) in atoms.iter().enumerate() {
            let x = 10.0 + res_idx as f64 * 3.8 + atom_idx as f64 * 0.3;
            let y = 20.0 + (atom_idx as f64 * 1.5).sin() * 2.0;
            let z = 30.0 + (res_idx as f64 * 0.7).cos() * 3.0;
            s.push_str(&format!(
                "ATOM {} {} A A 1 {} {} {:.3} {:.3} {:.3} {} 15.0 1.00 1\n",
                atom_name, res_name, seq_id, seq_id, x, y, z, elem
            ));
        }
    }
    s.push('#');
    s
}

fn sample_a3m_content(n_seqs: usize, seq_len: usize) -> String {
    let amino = b"ACDEFGHIKLMNPQRSTVWY-";
    let mut s = String::new();
    for i in 0..n_seqs {
        s.push_str(&format!(">seq_{} Tax=Species_{} TaxID={}\n", i, i % 20, i));
        for j in 0..seq_len {
            let idx = (i * 7 + j * 13) % amino.len();
            s.push(amino[idx] as char);
            // Sprinkle some lowercase insertions
            if j % 11 == 3 {
                s.push('a');
            }
        }
        s.push('\n');
    }
    s
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_cif_parsing(c: &mut Criterion) {
    let content = sample_cif_content();

    c.bench_function("cif_parse_10_residues", |b| {
        b.iter(|| {
            let array = CifParser::parse_str(black_box(&content)).unwrap();
            black_box(array.len());
        })
    });
}

fn bench_cif_parsing_large(c: &mut Criterion) {
    let mut s = String::from("data_LARGE\n#\nloop_\n");
    s.push_str("_atom_site.group_PDB\n");
    s.push_str("_atom_site.label_atom_id\n");
    s.push_str("_atom_site.label_comp_id\n");
    s.push_str("_atom_site.auth_asym_id\n");
    s.push_str("_atom_site.label_asym_id\n");
    s.push_str("_atom_site.label_entity_id\n");
    s.push_str("_atom_site.auth_seq_id\n");
    s.push_str("_atom_site.label_seq_id\n");
    s.push_str("_atom_site.Cartn_x\n");
    s.push_str("_atom_site.Cartn_y\n");
    s.push_str("_atom_site.Cartn_z\n");
    s.push_str("_atom_site.type_symbol\n");
    s.push_str("_atom_site.B_iso_or_equiv\n");
    s.push_str("_atom_site.occupancy\n");
    s.push_str("_atom_site.pdbx_PDB_model_num\n");

    let residue_names = ["ALA", "GLY", "VAL", "LEU", "ILE"];
    let backbone = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")];

    for res_idx in 0..500 {
        let res_name = residue_names[res_idx % residue_names.len()];
        let seq_id = res_idx + 1;
        let atoms = if res_name == "GLY" { &backbone[..4] } else { &backbone[..] };
        for (atom_idx, (atom_name, elem)) in atoms.iter().enumerate() {
            let x = (res_idx as f64) * 3.8 + atom_idx as f64;
            let y = (atom_idx as f64) * 1.2;
            let z = (res_idx as f64) * 0.5;
            s.push_str(&format!(
                "ATOM {} {} A A 1 {} {} {:.3} {:.3} {:.3} {} 10.0 1.00 1\n",
                atom_name, res_name, seq_id, seq_id, x, y, z, elem
            ));
        }
    }
    s.push('#');

    c.bench_function("cif_parse_500_residues", |b| {
        b.iter(|| {
            let array = CifParser::parse_str(black_box(&s)).unwrap();
            black_box(array.len());
        })
    });
}

fn bench_tokenization(c: &mut Criterion) {
    let content = sample_cif_content();
    let array = CifParser::parse_str(&content).unwrap();

    c.bench_function("tokenize_10_residues", |b| {
        b.iter(|| {
            let tokenizer = AtomArrayTokenizer::new(black_box(&array));
            let ta = tokenizer.get_token_array();
            black_box(ta.len());
        })
    });
}

fn bench_tokenization_large(c: &mut Criterion) {
    let mut s = String::from("data_TOK\n#\nloop_\n");
    s.push_str("_atom_site.group_PDB\n");
    s.push_str("_atom_site.label_atom_id\n");
    s.push_str("_atom_site.label_comp_id\n");
    s.push_str("_atom_site.auth_asym_id\n");
    s.push_str("_atom_site.label_asym_id\n");
    s.push_str("_atom_site.label_entity_id\n");
    s.push_str("_atom_site.auth_seq_id\n");
    s.push_str("_atom_site.label_seq_id\n");
    s.push_str("_atom_site.Cartn_x\n");
    s.push_str("_atom_site.Cartn_y\n");
    s.push_str("_atom_site.Cartn_z\n");
    s.push_str("_atom_site.type_symbol\n");
    s.push_str("_atom_site.B_iso_or_equiv\n");
    s.push_str("_atom_site.occupancy\n");
    s.push_str("_atom_site.pdbx_PDB_model_num\n");

    let names = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
                  "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
                  "THR", "TRP", "TYR", "VAL"];
    let backbone = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")];

    for res_idx in 0..200 {
        let res_name = names[res_idx % names.len()];
        let seq_id = res_idx + 1;
        let atoms = if res_name == "GLY" { &backbone[..4] } else { &backbone[..] };
        for (ai, (an, el)) in atoms.iter().enumerate() {
            s.push_str(&format!(
                "ATOM {} {} A A 1 {} {} {:.1} {:.1} {:.1} {} 10.0 1.00 1\n",
                an, res_name, seq_id, seq_id,
                res_idx as f64 + ai as f64, ai as f64, 0.0, el
            ));
        }
    }
    s.push('#');

    let array = CifParser::parse_str(&s).unwrap();

    c.bench_function("tokenize_200_residues", |b| {
        b.iter(|| {
            let tokenizer = AtomArrayTokenizer::new(black_box(&array));
            let ta = tokenizer.get_token_array();
            black_box(ta.len());
        })
    });
}

fn bench_a3m_parsing(c: &mut Criterion) {
    let content_small = sample_a3m_content(50, 100);
    c.bench_function("a3m_parse_50x100", |b| {
        b.iter(|| {
            let msa = parse_a3m(black_box(&content_small));
            black_box(msa.num_sequences());
        })
    });

    let content_large = sample_a3m_content(500, 300);
    c.bench_function("a3m_parse_500x300", |b| {
        b.iter(|| {
            let msa = parse_a3m(black_box(&content_large));
            black_box(msa.num_sequences());
        })
    });
}

fn bench_msa_profile(c: &mut Criterion) {
    let content = sample_a3m_content(200, 150);
    let msa = parse_a3m(&content);
    let seq_to_id = cattle_prod_core::constants::MSA_PROTEIN_SEQ_TO_ID.clone();

    c.bench_function("msa_profile_200x150", |b| {
        b.iter(|| {
            let profile = msa_profile(black_box(&msa), &seq_to_id, 32);
            black_box(profile.shape());
        })
    });
}

criterion_group!(
    benches,
    bench_cif_parsing,
    bench_cif_parsing_large,
    bench_tokenization,
    bench_tokenization_large,
    bench_a3m_parsing,
    bench_msa_profile,
);
criterion_main!(benches);
