"""
Simple Heterogeneous Probabilistic Graphical Model (PGM)

This example demonstrates a heterogeneous PGM with both discrete and continuous variables.
Scenario: Modeling student performance based on study hours and mood.

Network structure:
- Study Hours (Continuous) -> Exam Score (Continuous)
- Mood (Discrete: Good/Bad) -> Exam Score (Continuous)
- Study Hours (Continuous) -> Mood (Discrete)
"""

import numpy as np
from typing import Dict, List, Tuple


class HeterogeneousPGM:
    """A simple heterogeneous PGM with discrete and continuous nodes."""

    def __init__(self):
        # Define the network structure
        self.nodes = {
            'study_hours': 'continuous',  # Hours studied (0-10)
            'mood': 'discrete',           # Good (1) or Bad (0)
            'exam_score': 'continuous'    # Score (0-100)
        }

        # Store parameters
        self.parameters = {}
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize the conditional probability distributions."""

        # P(Study Hours) - Normal distribution
        self.parameters['study_hours'] = {
            'mean': 5.0,
            'std': 2.0
        }

        # P(Mood | Study Hours) - Logistic relationship
        # More study hours -> higher probability of good mood
        self.parameters['mood_given_study'] = {
            'threshold': 4.0,  # Hours where P(good mood) = 0.5
            'slope': 0.5       # Steepness of the relationship
        }

        # P(Exam Score | Study Hours, Mood)
        # Linear Gaussian: Score = base + coef1*hours + coef2*mood + noise
        self.parameters['score_given_study_mood'] = {
            'base': 50.0,
            'study_coefficient': 5.0,   # Each hour adds 5 points
            'mood_coefficient': 10.0,   # Good mood adds 10 points
            'noise_std': 5.0
        }

    def sample(self, n_samples: int = 1) -> List[Dict]:
        """Generate samples from the joint distribution."""
        samples = []

        for _ in range(n_samples):
            sample = {}

            # Sample Study Hours from prior
            study_hours = np.random.normal(
                self.parameters['study_hours']['mean'],
                self.parameters['study_hours']['std']
            )
            study_hours = np.clip(study_hours, 0, 10)  # Constrain to [0, 10]
            sample['study_hours'] = study_hours

            # Sample Mood given Study Hours (using sigmoid)
            mood_prob = self._compute_mood_probability(study_hours)
            mood = 1 if np.random.random() < mood_prob else 0
            sample['mood'] = 'Good' if mood == 1 else 'Bad'

            # Sample Exam Score given Study Hours and Mood
            params = self.parameters['score_given_study_mood']
            mean_score = (params['base'] +
                         params['study_coefficient'] * study_hours +
                         params['mood_coefficient'] * mood)

            score = np.random.normal(mean_score, params['noise_std'])
            score = np.clip(score, 0, 100)  # Constrain to [0, 100]
            sample['exam_score'] = score

            samples.append(sample)

        return samples

    def _compute_mood_probability(self, study_hours: float) -> float:
        """Compute P(Mood=Good | Study Hours) using sigmoid function."""
        params = self.parameters['mood_given_study']
        z = params['slope'] * (study_hours - params['threshold'])
        return 1 / (1 + np.exp(-z))

    def predict_score(self, study_hours: float, mood: str) -> Tuple[float, float]:
        """
        Predict exam score given study hours and mood.

        Returns:
            (mean_score, std_score): Expected score and standard deviation
        """
        params = self.parameters['score_given_study_mood']
        mood_value = 1 if mood.lower() == 'good' else 0

        mean_score = (params['base'] +
                     params['study_coefficient'] * study_hours +
                     params['mood_coefficient'] * mood_value)

        return mean_score, params['noise_std']

    def infer_mood_probability(self, study_hours: float) -> Dict[str, float]:
        """Infer the probability of mood states given study hours."""
        prob_good = self._compute_mood_probability(study_hours)
        return {
            'Good': prob_good,
            'Bad': 1 - prob_good
        }

    def print_network_info(self):
        """Print information about the network structure."""
        print("=" * 60)
        print("HETEROGENEOUS PGM: Student Performance Model")
        print("=" * 60)
        print("\nNodes:")
        for node, node_type in self.nodes.items():
            print(f"  - {node}: {node_type}")

        print("\nEdges (Dependencies):")
        print("  - study_hours → mood")
        print("  - study_hours → exam_score")
        print("  - mood → exam_score")

        print("\nParameters:")
        print(f"  Study Hours ~ Normal(μ={self.parameters['study_hours']['mean']}, "
              f"σ={self.parameters['study_hours']['std']})")
        print(f"  Mood | Study Hours ~ Bernoulli(sigmoid(study_hours))")
        print(f"  Exam Score | Study Hours, Mood ~ Linear Gaussian")
        print("=" * 60)


def main():
    """Demonstrate the heterogeneous PGM."""

    # Create the model
    print("\n🎓 Creating Heterogeneous PGM...\n")
    model = HeterogeneousPGM()
    model.print_network_info()

    # Generate samples
    print("\n📊 Generating 10 samples from the joint distribution:\n")
    samples = model.sample(n_samples=10)

    print(f"{'Study Hours':<15} {'Mood':<10} {'Exam Score':<12}")
    print("-" * 40)
    for sample in samples:
        print(f"{sample['study_hours']:>12.2f}   {sample['mood']:<10} {sample['exam_score']:>10.2f}")

    # Inference examples
    print("\n🔍 Inference Examples:\n")

    # Example 1: Predict mood probability given study hours
    study_hours = 6.0
    mood_probs = model.infer_mood_probability(study_hours)
    print(f"Given {study_hours} hours of study:")
    print(f"  P(Mood = Good) = {mood_probs['Good']:.3f}")
    print(f"  P(Mood = Bad) = {mood_probs['Bad']:.3f}")

    # Example 2: Predict exam score
    mean_score, std_score = model.predict_score(study_hours=7.0, mood='Good')
    print(f"\nGiven 7 hours of study and Good mood:")
    print(f"  Expected Exam Score = {mean_score:.2f} ± {std_score:.2f}")

    mean_score, std_score = model.predict_score(study_hours=3.0, mood='Bad')
    print(f"\nGiven 3 hours of study and Bad mood:")
    print(f"  Expected Exam Score = {mean_score:.2f} ± {std_score:.2f}")

    # Statistical analysis
    print("\n📈 Statistical Analysis (1000 samples):\n")
    large_samples = model.sample(n_samples=1000)

    avg_study = np.mean([s['study_hours'] for s in large_samples])
    avg_score = np.mean([s['exam_score'] for s in large_samples])
    pct_good_mood = np.mean([1 if s['mood'] == 'Good' else 0 for s in large_samples])

    print(f"Average Study Hours: {avg_study:.2f}")
    print(f"Average Exam Score: {avg_score:.2f}")
    print(f"Percentage with Good Mood: {pct_good_mood * 100:.1f}%")

    # Correlation analysis
    study_hours_arr = np.array([s['study_hours'] for s in large_samples])
    scores_arr = np.array([s['exam_score'] for s in large_samples])
    correlation = np.corrcoef(study_hours_arr, scores_arr)[0, 1]
    print(f"\nCorrelation between Study Hours and Exam Score: {correlation:.3f}")


if __name__ == "__main__":
    main()
