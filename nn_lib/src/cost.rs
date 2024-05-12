enum Cost {
    CrossEntropy,
}

impl Cost {
    fn evaluate(&self) -> f64 {
        match self {
            Self::CrossEntropy => {
                todo!()
            }
        }
    }
}
