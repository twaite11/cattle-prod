use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MolType {
    Protein,
    Rna,
    Dna,
    Ligand,
    Branched,
    Macrolide,
    NonPolymer,
    Unknown,
}

impl MolType {
    pub fn is_polymer(&self) -> bool {
        matches!(self, Self::Protein | Self::Rna | Self::Dna)
    }

    pub fn is_ligand_type(&self) -> bool {
        matches!(self, Self::Ligand | Self::Branched | Self::Macrolide | Self::NonPolymer)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChainType {
    PolypeptideL,
    Polyribonucleotide,
    Polydeoxyribonucleotide,
    Branched,
    Macrolide,
    NonPolymer,
}

impl ChainType {
    pub const PROTEIN: &'static str = "polypeptide(L)";
    pub const RNA: &'static str = "polyribonucleotide";
    pub const DNA: &'static str = "polydeoxyribonucleotide";
    pub const BRANCHED: &'static str = "branched";
    pub const MACROLIDE: &'static str = "macrolide";
    pub const NON_POLYMER: &'static str = "non-polymer";

    pub fn from_str_label(s: &str) -> Option<Self> {
        match s {
            "polypeptide(L)" => Some(Self::PolypeptideL),
            "polyribonucleotide" => Some(Self::Polyribonucleotide),
            "polydeoxyribonucleotide" => Some(Self::Polydeoxyribonucleotide),
            "branched" => Some(Self::Branched),
            "macrolide" => Some(Self::Macrolide),
            "non-polymer" => Some(Self::NonPolymer),
            _ => None,
        }
    }

    pub fn is_standard_polymer(&self) -> bool {
        matches!(
            self,
            Self::PolypeptideL | Self::Polyribonucleotide | Self::Polydeoxyribonucleotide
        )
    }

    pub fn is_ligand_type(&self) -> bool {
        matches!(self, Self::Branched | Self::Macrolide | Self::NonPolymer)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PolypeptideL => Self::PROTEIN,
            Self::Polyribonucleotide => Self::RNA,
            Self::Polydeoxyribonucleotide => Self::DNA,
            Self::Branched => Self::BRANCHED,
            Self::Macrolide => Self::MACROLIDE,
            Self::NonPolymer => Self::NON_POLYMER,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dtype {
    Bf16,
    Fp16,
    Fp32,
}

impl Dtype {
    pub fn from_str_label(s: &str) -> Self {
        match s {
            "bf16" => Self::Bf16,
            "fp16" => Self::Fp16,
            "fp32" | "float32" => Self::Fp32,
            _ => Self::Bf16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TriangleKernel {
    Torch,
    Triattention,
    Cuequivariance,
    Deepspeed,
}

impl TriangleKernel {
    pub fn from_str_label(s: &str) -> Self {
        match s {
            "torch" => Self::Torch,
            "triattention" => Self::Triattention,
            "cuequivariance" => Self::Cuequivariance,
            "deepspeed" => Self::Deepspeed,
            _ => Self::Torch,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvalChainInterface {
    IntraLigand,
    IntraDna,
    IntraRna,
    IntraProt,
    LigandProt,
    RnaProt,
    DnaProt,
    ProtProt,
    AntibodyAntigen,
    Antibody,
}

impl EvalChainInterface {
    pub fn all() -> &'static [Self] {
        &[
            Self::IntraLigand,
            Self::IntraDna,
            Self::IntraRna,
            Self::IntraProt,
            Self::LigandProt,
            Self::RnaProt,
            Self::DnaProt,
            Self::ProtProt,
            Self::AntibodyAntigen,
            Self::Antibody,
        ]
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::IntraLigand => "intra_ligand",
            Self::IntraDna => "intra_dna",
            Self::IntraRna => "intra_rna",
            Self::IntraProt => "intra_prot",
            Self::LigandProt => "ligand_prot",
            Self::RnaProt => "rna_prot",
            Self::DnaProt => "dna_prot",
            Self::ProtProt => "prot_prot",
            Self::AntibodyAntigen => "antibody_antigen",
            Self::Antibody => "antibody",
        }
    }
}
