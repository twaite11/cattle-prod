use std::collections::HashMap;

use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Raw MSA representation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct RawMsa {
    pub descriptions: Vec<String>,
    pub sequences: Vec<String>,
    pub deletion_matrix: Vec<Vec<i32>>,
}

impl RawMsa {
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }

    pub fn sequence_length(&self) -> usize {
        self.sequences.first().map_or(0, |s| s.len())
    }
}

// ---------------------------------------------------------------------------
// A3M parser
// ---------------------------------------------------------------------------

/// Parse A3M-formatted MSA text.
///
/// In A3M format:
///  - Lines starting with `>` are description headers.
///  - Uppercase letters and `-` are alignment columns.
///  - Lowercase letters represent insertions (not aligned).
///
/// Returns a `RawMsa` with aligned `sequences` (uppercase + '-') and a
/// `deletion_matrix` counting insertions between each aligned column.
pub fn parse_a3m(content: &str) -> RawMsa {
    let mut descs = Vec::new();
    let mut seqs = Vec::new();
    let mut current_seq = String::new();
    let mut current_desc = String::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('>') {
            if !current_seq.is_empty() || !current_desc.is_empty() {
                descs.push(current_desc.clone());
                seqs.push(std::mem::take(&mut current_seq));
            }
            current_desc = trimmed[1..].to_string();
        } else {
            current_seq.push_str(trimmed);
        }
    }
    if !current_seq.is_empty() || !current_desc.is_empty() {
        descs.push(current_desc);
        seqs.push(current_seq);
    }

    // Separate aligned residues from insertions.
    let mut aligned_seqs = Vec::with_capacity(seqs.len());
    let mut del_matrix = Vec::with_capacity(seqs.len());

    for raw_seq in &seqs {
        let mut aligned = String::new();
        let mut deletions: Vec<i32> = Vec::new();
        let mut insertion_count: i32 = 0;

        for ch in raw_seq.chars() {
            if ch.is_ascii_uppercase() || ch == '-' {
                // aligned column
                deletions.push(insertion_count);
                insertion_count = 0;
                aligned.push(ch);
            } else if ch.is_ascii_lowercase() {
                // insertion
                insertion_count += 1;
            }
            // skip other characters
        }

        if insertion_count > 0 {
            if let Some(last) = deletions.last_mut() {
                *last += insertion_count;
            }
        }

        aligned_seqs.push(aligned);
        del_matrix.push(deletions);
    }

    RawMsa {
        descriptions: descs,
        sequences: aligned_seqs,
        deletion_matrix: del_matrix,
    }
}

// ---------------------------------------------------------------------------
// MSA features
// ---------------------------------------------------------------------------

/// Compute a position-frequency profile from the MSA.
///
/// Returns an `[L, num_classes]` array where L is the alignment length.
/// Each aligned character is mapped through `seq_to_id`; unknown characters
/// are skipped.
pub fn msa_profile(
    msa: &RawMsa,
    seq_to_id: &HashMap<char, u8>,
    num_classes: usize,
) -> Array2<f32> {
    let seqlen = msa.sequence_length();
    if seqlen == 0 || msa.sequences.is_empty() {
        return Array2::zeros((0, num_classes));
    }

    let mut counts = Array2::<f32>::zeros((seqlen, num_classes));

    for seq in &msa.sequences {
        for (pos, ch) in seq.chars().enumerate() {
            if pos >= seqlen {
                break;
            }
            if let Some(&id) = seq_to_id.get(&ch) {
                let cls = id as usize;
                if cls < num_classes {
                    counts[[pos, cls]] += 1.0;
                }
            }
        }
    }

    let n = msa.num_sequences() as f32;
    if n > 0.0 {
        counts /= n;
    }
    counts
}

/// Mean deletion value per alignment column, averaged over all sequences.
pub fn deletion_mean(msa: &RawMsa) -> Array1<f32> {
    let seqlen = msa.sequence_length();
    if seqlen == 0 || msa.deletion_matrix.is_empty() {
        return Array1::zeros(0);
    }

    let mut sums = vec![0.0f32; seqlen];
    let mut count = 0usize;
    for row in &msa.deletion_matrix {
        for (j, &v) in row.iter().enumerate() {
            if j < seqlen {
                sums[j] += v as f32;
            }
        }
        count += 1;
    }

    if count > 0 {
        let cf = count as f32;
        for s in &mut sums {
            *s /= cf;
        }
    }

    Array1::from_vec(sums)
}

// ---------------------------------------------------------------------------
// MSA pairing by species
// ---------------------------------------------------------------------------

/// Extracts an organism / taxonomy identifier from an A3M description line.
///
/// Looks for `Tax=<value>` or `OX=<value>` patterns commonly found in
/// UniRef FASTA headers. Returns `None` when no identifier is found.
fn extract_species(desc: &str) -> Option<String> {
    // Try "Tax=" first (UniProt style)
    if let Some(start) = desc.find("Tax=") {
        let rest = &desc[start + 4..];
        let end = rest.find(" TaxID=").or_else(|| rest.find('\t')).unwrap_or(rest.len());
        let species = rest[..end].trim().to_string();
        if !species.is_empty() {
            return Some(species);
        }
    }
    // Try "OX=" (another common pattern)
    if let Some(start) = desc.find("OX=") {
        let rest = &desc[start + 3..];
        let end = rest
            .find(|c: char| c.is_whitespace())
            .unwrap_or(rest.len());
        let ox = rest[..end].trim().to_string();
        if !ox.is_empty() {
            return Some(ox);
        }
    }
    None
}

/// Engine for pairing MSA rows across multiple chains by shared species.
///
/// Given MSAs for several chains, it identifies rows that share the same
/// organism and pairs them so that each row in the paired MSA represents
/// one organism across all chains.
pub struct MsaPairingEngine;

impl MsaPairingEngine {
    /// Pair MSAs from multiple chains by organism identifier.
    ///
    /// For each chain we have a `RawMsa`. The result is one `RawMsa` per
    /// chain with the same number of rows, where row *k* corresponds to the
    /// same species across all chains. Chains that do not have a hit for a
    /// given species receive a gap-only row.
    pub fn pair_by_species(msas: &[RawMsa]) -> Vec<RawMsa> {
        if msas.is_empty() {
            return Vec::new();
        }

        // Build species → row-index maps per chain.
        let chain_species: Vec<HashMap<String, Vec<usize>>> = msas
            .iter()
            .map(|msa| {
                let mut map: HashMap<String, Vec<usize>> = HashMap::new();
                for (i, desc) in msa.descriptions.iter().enumerate() {
                    if i == 0 {
                        continue; // skip query row
                    }
                    if let Some(sp) = extract_species(desc) {
                        map.entry(sp).or_default().push(i);
                    }
                }
                map
            })
            .collect();

        // Collect all species appearing in at least two chains.
        let mut species_counts: HashMap<&str, usize> = HashMap::new();
        for cs in &chain_species {
            for sp in cs.keys() {
                *species_counts.entry(sp.as_str()).or_insert(0) += 1;
            }
        }
        let shared_species: Vec<&str> = species_counts
            .into_iter()
            .filter(|&(_, c)| c >= 2)
            .map(|(sp, _)| sp)
            .collect();

        // Build paired MSAs: first row is always the query sequence.
        let n_chains = msas.len();
        let mut paired: Vec<RawMsa> = msas
            .iter()
            .map(|msa| {
                let mut p = RawMsa::default();
                if !msa.sequences.is_empty() {
                    p.descriptions.push(msa.descriptions[0].clone());
                    p.sequences.push(msa.sequences[0].clone());
                    p.deletion_matrix.push(
                        msa.deletion_matrix
                            .first()
                            .cloned()
                            .unwrap_or_default(),
                    );
                }
                p
            })
            .collect();

        for sp in &shared_species {
            for ci in 0..n_chains {
                let seqlen = msas[ci].sequence_length();
                if let Some(rows) = chain_species[ci].get(*sp) {
                    let ri = rows[0]; // take first hit for this species
                    paired[ci]
                        .descriptions
                        .push(msas[ci].descriptions[ri].clone());
                    paired[ci]
                        .sequences
                        .push(msas[ci].sequences[ri].clone());
                    paired[ci]
                        .deletion_matrix
                        .push(msas[ci].deletion_matrix[ri].clone());
                } else {
                    // gap-only row
                    paired[ci].descriptions.push(String::new());
                    paired[ci]
                        .sequences
                        .push("-".repeat(seqlen));
                    paired[ci]
                        .deletion_matrix
                        .push(vec![0i32; seqlen]);
                }
            }
        }

        paired
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_a3m_basic() {
        let content = "\
>query
AcGtM
>hit1 Tax=Homo sapiens TaxID=9606
A-Gtm
";
        let msa = parse_a3m(content);
        assert_eq!(msa.num_sequences(), 2);
        // query: uppercase A, lowercase c -> insertion, uppercase G,
        //        lowercase t -> insertion, uppercase M
        assert_eq!(&msa.sequences[0], "AGM");
        assert_eq!(msa.deletion_matrix[0], vec![0, 1, 1]);
        // hit1: A, -, G, lowercase t -> insertion, lowercase m -> insertion
        assert_eq!(&msa.sequences[1], "A-G");
        assert_eq!(msa.deletion_matrix[1], vec![0, 0, 2]);
    }

    #[test]
    fn test_extract_species() {
        let desc = "UniRef100_A0A0 n=1 Tax=Homo sapiens TaxID=9606 RepID=FOO";
        assert_eq!(extract_species(desc), Some("Homo sapiens".to_string()));
    }
}
