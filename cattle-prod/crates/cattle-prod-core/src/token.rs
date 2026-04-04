use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Token {
    pub value: u32,
    pub atom_indices: Vec<usize>,
    pub atom_names: Vec<String>,
    pub annotations: HashMap<String, AnnotValue>,
}

#[derive(Debug, Clone)]
pub enum AnnotValue {
    Usize(usize),
    I64(i64),
    F64(f64),
    String(String),
    VecUsize(Vec<usize>),
    VecString(Vec<String>),
}

impl AnnotValue {
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Self::Usize(v) => Some(*v),
            Self::I64(v) => Some(*v as usize),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(v) => Some(*v),
            Self::I64(v) => Some(*v as f64),
            Self::Usize(v) => Some(*v as f64),
            _ => None,
        }
    }
}

impl Token {
    pub fn new(value: u32) -> Self {
        Self {
            value,
            atom_indices: Vec::new(),
            atom_names: Vec::new(),
            annotations: HashMap::new(),
        }
    }

    pub fn with_atoms(value: u32, atom_indices: Vec<usize>, atom_names: Vec<String>) -> Self {
        Self {
            value,
            atom_indices,
            atom_names,
            annotations: HashMap::new(),
        }
    }

    pub fn set_annotation(&mut self, key: &str, value: AnnotValue) {
        self.annotations.insert(key.to_string(), value);
    }

    pub fn get_annotation(&self, key: &str) -> Option<&AnnotValue> {
        self.annotations.get(key)
    }

    pub fn centre_atom_index(&self) -> Option<usize> {
        self.annotations
            .get("centre_atom_index")
            .and_then(|v| v.as_usize())
    }
}

#[derive(Debug, Clone)]
pub struct TokenArray {
    pub tokens: Vec<Token>,
}

impl TokenArray {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&Token> {
        self.tokens.get(index)
    }

    pub fn select(&self, indices: &[usize]) -> Self {
        let tokens = indices
            .iter()
            .filter_map(|&i| self.tokens.get(i).cloned())
            .collect();
        Self { tokens }
    }

    pub fn values(&self) -> Vec<u32> {
        self.tokens.iter().map(|t| t.value).collect()
    }

    pub fn get_annotation_vec(&self, key: &str) -> Vec<Option<&AnnotValue>> {
        self.tokens
            .iter()
            .map(|t| t.get_annotation(key))
            .collect()
    }

    pub fn set_annotation_vec(&mut self, key: &str, values: Vec<AnnotValue>) {
        assert_eq!(
            values.len(),
            self.tokens.len(),
            "annotation length must match token count"
        );
        for (token, val) in self.tokens.iter_mut().zip(values) {
            token.set_annotation(key, val);
        }
    }

    pub fn set_centre_atom_indices(&mut self, indices: &[usize]) {
        assert_eq!(
            indices.len(),
            self.tokens.len(),
            "centre atom index count must match token count"
        );
        for (token, &idx) in self.tokens.iter_mut().zip(indices) {
            token.set_annotation("centre_atom_index", AnnotValue::Usize(idx));
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Token> {
        self.tokens.iter()
    }
}

impl<'a> IntoIterator for &'a TokenArray {
    type Item = &'a Token;
    type IntoIter = std::slice::Iter<'a, Token>;

    fn into_iter(self) -> Self::IntoIter {
        self.tokens.iter()
    }
}
