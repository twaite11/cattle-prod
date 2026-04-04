use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct ConfidenceSummary {
    pub plddt: f64,
    pub ptm: f64,
    pub iptm: f64,
    pub ranking_score: f64,
    pub chain_ptm: Vec<f64>,
    pub chain_plddt: Vec<f64>,
}

pub struct DataDumper {
    base_dir: PathBuf,
    need_atom_confidence: bool,
    sorted_by_ranking_score: bool,
}

impl DataDumper {
    pub fn new(
        base_dir: PathBuf,
        need_atom_confidence: bool,
        sorted_by_ranking_score: bool,
    ) -> Self {
        Self {
            base_dir,
            need_atom_confidence,
            sorted_by_ranking_score,
        }
    }

    /// Write predicted structure as mmCIF file.
    pub fn write_cif(
        &self,
        coords: &[[f32; 3]],
        atom_names: &[String],
        res_names: &[String],
        chain_ids: &[String],
        res_ids: &[i32],
        elements: &[String],
        b_factors: Option<&[f32]>,
        output_path: &Path,
    ) -> anyhow::Result<()> {
        let n = coords.len();
        anyhow::ensure!(
            atom_names.len() == n
                && res_names.len() == n
                && chain_ids.len() == n
                && res_ids.len() == n
                && elements.len() == n,
            "All atom-level arrays must have the same length (got {n} coords)"
        );

        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let stem = output_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("prediction");

        let mut f = fs::File::create(output_path)?;

        // ── data_ block header ──────────────────────────────────────
        writeln!(f, "data_{stem}")?;
        writeln!(f, "#")?;

        // ── _atom_site loop ─────────────────────────────────────────
        writeln!(f, "loop_")?;
        writeln!(f, "_atom_site.group_PDB")?;
        writeln!(f, "_atom_site.id")?;
        writeln!(f, "_atom_site.type_symbol")?;
        writeln!(f, "_atom_site.label_atom_id")?;
        writeln!(f, "_atom_site.label_comp_id")?;
        writeln!(f, "_atom_site.label_asym_id")?;
        writeln!(f, "_atom_site.label_seq_id")?;
        writeln!(f, "_atom_site.auth_asym_id")?;
        writeln!(f, "_atom_site.auth_seq_id")?;
        writeln!(f, "_atom_site.Cartn_x")?;
        writeln!(f, "_atom_site.Cartn_y")?;
        writeln!(f, "_atom_site.Cartn_z")?;
        writeln!(f, "_atom_site.occupancy")?;
        writeln!(f, "_atom_site.B_iso_or_equiv")?;
        writeln!(f, "_atom_site.pdbx_PDB_model_num")?;

        for i in 0..n {
            let group = "ATOM";
            let serial = i + 1;
            let element = &elements[i];
            let atom_name = &atom_names[i];
            let comp_id = &res_names[i];
            let asym_id = &chain_ids[i];
            let seq_id = res_ids[i];
            let x = coords[i][0];
            let y = coords[i][1];
            let z = coords[i][2];
            let occ = 1.00_f32;
            let b = b_factors.map_or(0.00_f32, |bf| bf[i]);

            writeln!(
                f,
                "{group:<6}{serial:>5} {element:<2} {atom_name:<4} {comp_id:<3} \
                 {asym_id:<2} {seq_id:>4} {asym_id:<2} {seq_id:>4} \
                 {x:>8.3} {y:>8.3} {z:>8.3} {occ:>6.2} {b:>6.2} 1",
            )?;
        }

        writeln!(f, "#")?;
        writeln!(f, "_entry.id  {stem}")?;
        writeln!(f, "#")?;

        log::debug!("Wrote mmCIF with {n} atoms to {}", output_path.display());
        Ok(())
    }

    /// Write confidence summary as JSON.
    pub fn write_confidence_json(
        &self,
        summary: &ConfidenceSummary,
        output_path: &Path,
    ) -> anyhow::Result<()> {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(summary)?;
        fs::write(output_path, json)?;
        log::debug!(
            "Wrote confidence JSON to {}",
            output_path.display()
        );
        Ok(())
    }

    /// Main dump function: write all outputs for a single prediction sample.
    ///
    /// Directory layout:
    /// ```text
    /// {base_dir}/{dataset_name}/{pdb_id}/
    ///     seed_{seed}/
    ///         sample_{sample_idx}/
    ///             rank_{rank}_prediction.cif
    ///             rank_{rank}_confidence.json
    ///             rank_{rank}_plddt.json          (if need_atom_confidence)
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn dump(
        &self,
        dataset_name: &str,
        pdb_id: &str,
        seed: u64,
        sample_idx: usize,
        rank: usize,
        coords: &[[f32; 3]],
        atom_names: &[String],
        res_names: &[String],
        chain_ids: &[String],
        res_ids: &[i32],
        elements: &[String],
        confidence: &ConfidenceSummary,
        plddt_per_atom: Option<&[f32]>,
    ) -> anyhow::Result<PathBuf> {
        let sample_dir = self
            .base_dir
            .join(dataset_name)
            .join(pdb_id)
            .join(format!("seed_{seed}"))
            .join(format!("sample_{sample_idx}"));
        fs::create_dir_all(&sample_dir)?;

        let prefix = if self.sorted_by_ranking_score {
            format!("rank_{rank:03}")
        } else {
            format!("sample_{sample_idx}")
        };

        // ── CIF ─────────────────────────────────────────────────────
        let cif_path = sample_dir.join(format!("{prefix}_prediction.cif"));
        self.write_cif(
            coords,
            atom_names,
            res_names,
            chain_ids,
            res_ids,
            elements,
            plddt_per_atom,
            &cif_path,
        )?;

        // ── Confidence summary ──────────────────────────────────────
        let conf_path = sample_dir.join(format!("{prefix}_confidence.json"));
        self.write_confidence_json(confidence, &conf_path)?;

        // ── Per-atom pLDDT (optional) ───────────────────────────────
        if self.need_atom_confidence {
            if let Some(plddt) = plddt_per_atom {
                let plddt_path = sample_dir.join(format!("{prefix}_plddt.json"));
                let json = serde_json::to_string_pretty(plddt)?;
                fs::write(&plddt_path, json)?;
                log::debug!("Wrote per-atom pLDDT to {}", plddt_path.display());
            }
        }

        Ok(sample_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_cif_round_trip() {
        let dir = std::env::temp_dir().join("cattle_prod_test_cif");
        let _ = fs::remove_dir_all(&dir);
        let dumper = DataDumper::new(dir.clone(), false, true);

        let coords = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let atom_names = vec!["CA".to_string(), "C".to_string()];
        let res_names = vec!["ALA".to_string(), "ALA".to_string()];
        let chain_ids = vec!["A".to_string(), "A".to_string()];
        let res_ids = vec![1, 1];
        let elements = vec!["C".to_string(), "C".to_string()];
        let b_factors = vec![50.0_f32, 60.0];

        let out_path = dir.join("test.cif");
        dumper
            .write_cif(
                &coords,
                &atom_names,
                &res_names,
                &chain_ids,
                &res_ids,
                &elements,
                Some(&b_factors),
                &out_path,
            )
            .unwrap();

        let content = fs::read_to_string(&out_path).unwrap();
        assert!(content.contains("data_test"));
        assert!(content.contains("_atom_site.Cartn_x"));
        assert!(content.contains("ATOM"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_confidence_json() {
        let dir = std::env::temp_dir().join("cattle_prod_test_conf");
        let _ = fs::remove_dir_all(&dir);
        let dumper = DataDumper::new(dir.clone(), false, true);

        let summary = ConfidenceSummary {
            plddt: 72.5,
            ptm: 0.85,
            iptm: 0.78,
            ranking_score: 0.81,
            chain_ptm: vec![0.85],
            chain_plddt: vec![72.5],
        };

        let out_path = dir.join("conf.json");
        dumper.write_confidence_json(&summary, &out_path).unwrap();

        let content = fs::read_to_string(&out_path).unwrap();
        let parsed: ConfidenceSummary = serde_json::from_str(&content).unwrap();
        assert!((parsed.plddt - 72.5).abs() < 1e-6);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_dump_creates_directory_tree() {
        let dir = std::env::temp_dir().join("cattle_prod_test_dump");
        let _ = fs::remove_dir_all(&dir);
        let dumper = DataDumper::new(dir.clone(), true, true);

        let coords = vec![[1.0, 2.0, 3.0]];
        let names = vec!["CA".to_string()];
        let res = vec!["ALA".to_string()];
        let chains = vec!["A".to_string()];
        let ids = vec![1];
        let elems = vec!["C".to_string()];
        let plddt = vec![80.0_f32];
        let conf = ConfidenceSummary {
            plddt: 80.0,
            ptm: 0.9,
            iptm: 0.8,
            ranking_score: 0.85,
            chain_ptm: vec![0.9],
            chain_plddt: vec![80.0],
        };

        let result = dumper
            .dump("ds", "pdb1", 42, 0, 0, &coords, &names, &res, &chains, &ids, &elems, &conf, Some(&plddt))
            .unwrap();

        assert!(result.exists());
        assert!(result.join("rank_000_prediction.cif").exists());
        assert!(result.join("rank_000_confidence.json").exists());
        assert!(result.join("rank_000_plddt.json").exists());

        let _ = fs::remove_dir_all(&dir);
    }
}
