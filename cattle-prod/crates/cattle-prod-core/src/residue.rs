use std::collections::HashMap;
use std::sync::LazyLock;

pub const NUM_STD_RESIDUES: usize = 32;
pub const RNA_START_INDEX: usize = 21;
pub const DNA_START_INDEX: usize = 26;

pub static PRO_STD_RESIDUES: LazyLock<HashMap<&'static str, u8>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("ALA", 0);
    m.insert("ARG", 1);
    m.insert("ASN", 2);
    m.insert("ASP", 3);
    m.insert("CYS", 4);
    m.insert("GLN", 5);
    m.insert("GLU", 6);
    m.insert("GLY", 7);
    m.insert("HIS", 8);
    m.insert("ILE", 9);
    m.insert("LEU", 10);
    m.insert("LYS", 11);
    m.insert("MET", 12);
    m.insert("PHE", 13);
    m.insert("PRO", 14);
    m.insert("SER", 15);
    m.insert("THR", 16);
    m.insert("TRP", 17);
    m.insert("TYR", 18);
    m.insert("VAL", 19);
    m.insert("UNK", 20);
    m
});

pub static RNA_STD_RESIDUES: LazyLock<HashMap<&'static str, u8>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("A", 21);
    m.insert("G", 22);
    m.insert("C", 23);
    m.insert("U", 24);
    m.insert("N", 25);
    m
});

pub static DNA_STD_RESIDUES: LazyLock<HashMap<&'static str, u8>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("DA", 26);
    m.insert("DG", 27);
    m.insert("DC", 28);
    m.insert("DT", 29);
    m.insert("DN", 30);
    m
});

pub static STD_RESIDUES: LazyLock<HashMap<&'static str, u8>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.extend(PRO_STD_RESIDUES.iter());
    m.extend(RNA_STD_RESIDUES.iter());
    m.extend(DNA_STD_RESIDUES.iter());
    m
});

pub static STD_RESIDUES_WITH_GAP: LazyLock<HashMap<&'static str, u8>> = LazyLock::new(|| {
    let mut m = STD_RESIDUES.clone();
    m.insert("-", 31);
    m
});

pub static STD_RESIDUES_WITH_GAP_ID_TO_NAME: LazyLock<HashMap<u8, &'static str>> =
    LazyLock::new(|| STD_RESIDUES_WITH_GAP.iter().map(|(&k, &v)| (v, k)).collect());

pub const PROTEIN_TYPES_ONE_LETTER: [char; 20] = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
    'Y', 'V',
];

pub static PROT_STD_RESIDUES_ONE_TO_THREE: LazyLock<HashMap<char, &'static str>> =
    LazyLock::new(|| {
        let mut m = HashMap::new();
        m.insert('A', "ALA");
        m.insert('R', "ARG");
        m.insert('N', "ASN");
        m.insert('D', "ASP");
        m.insert('C', "CYS");
        m.insert('Q', "GLN");
        m.insert('E', "GLU");
        m.insert('G', "GLY");
        m.insert('H', "HIS");
        m.insert('I', "ILE");
        m.insert('L', "LEU");
        m.insert('K', "LYS");
        m.insert('M', "MET");
        m.insert('F', "PHE");
        m.insert('P', "PRO");
        m.insert('S', "SER");
        m.insert('T', "THR");
        m.insert('W', "TRP");
        m.insert('Y', "TYR");
        m.insert('V', "VAL");
        m.insert('X', "UNK");
        m.insert('-', "-");
        m
    });

pub static PROTEIN_COMMON_ONE_TO_THREE: LazyLock<HashMap<char, &'static str>> =
    LazyLock::new(|| {
        let mut m = HashMap::new();
        m.insert('A', "ALA");
        m.insert('R', "ARG");
        m.insert('N', "ASN");
        m.insert('D', "ASP");
        m.insert('C', "CYS");
        m.insert('Q', "GLN");
        m.insert('E', "GLU");
        m.insert('G', "GLY");
        m.insert('H', "HIS");
        m.insert('I', "ILE");
        m.insert('L', "LEU");
        m.insert('K', "LYS");
        m.insert('M', "MET");
        m.insert('F', "PHE");
        m.insert('P', "PRO");
        m.insert('S', "SER");
        m.insert('T', "THR");
        m.insert('W', "TRP");
        m.insert('Y', "TYR");
        m.insert('V', "VAL");
        m
    });

pub static MMCIF_RESTYPE_1TO3: LazyLock<HashMap<char, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert('A', "ALA");
    m.insert('R', "ARG");
    m.insert('N', "ASN");
    m.insert('D', "ASP");
    m.insert('C', "CYS");
    m.insert('Q', "GLN");
    m.insert('E', "GLU");
    m.insert('G', "GLY");
    m.insert('H', "HIS");
    m.insert('I', "ILE");
    m.insert('L', "LEU");
    m.insert('K', "LYS");
    m.insert('M', "MET");
    m.insert('F', "PHE");
    m.insert('P', "PRO");
    m.insert('S', "SER");
    m.insert('T', "THR");
    m.insert('W', "TRP");
    m.insert('Y', "TYR");
    m.insert('V', "VAL");
    m.insert('B', "ASX");
    m.insert('Z', "GLX");
    m
});

pub static MMCIF_RESTYPE_3TO1: LazyLock<HashMap<&'static str, char>> =
    LazyLock::new(|| MMCIF_RESTYPE_1TO3.iter().map(|(&k, &v)| (v, k)).collect());

pub const RNA_TYPES: [&str; 4] = ["A", "G", "C", "U"];
pub const DNA_TYPES: [&str; 4] = ["DA", "DG", "DC", "DT"];
