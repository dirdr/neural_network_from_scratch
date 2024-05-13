pub enum Cost {
    /// The use case for CrossEntropy, is for our classification nn, taking
    /// softmax outputs and calcualting loss.
    CrossEntropy,
}

impl Cost {
    /// Calculate the classification cost for a predicted probability, which is in [0; 1]
    pub fn classification_cost(&self, predicted_probability: f64) -> f64 {
        match self {
            Self::CrossEntropy => -f64::ln(predicted_probability),
        }
    }
}
