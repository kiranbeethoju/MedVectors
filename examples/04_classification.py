"""
MedVectors Classification Example
================================

This example demonstrates using MedVectors embeddings for classification tasks.

Use Cases:
- Disease category prediction
- Medical document classification
- Symptom-based triage
- Medical specialty routing
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Dict, Tuple
import json


class MedVectorsClassifier:
    """Text classifier using MedVectors embeddings."""

    def __init__(
        self,
        model_name: str = "abhinand/MedVectors-base-v0.1",
        device: str = None,
        max_length: int = 512
    ):
        """
        Initialize MedVectors classifier.

        Args:
            model_name: Hugging Face model name
            device: Device to use (cuda, mps, cpu). Auto-detects if None
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.classifier = None
        self.label_encoder = None

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Encode texts into embeddings."""
        all_embeddings = []

        for batch_texts in [texts[i:i + 32] for i in range(0, len(texts), 32)]:
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Mean pooling
            embeddings = self._mean_pooling(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )

            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling using attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def train(
        self,
        texts: List[str],
        labels: List[str],
        classifier_type: str = "logistic",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Train a classifier using MedVectors embeddings.

        Args:
            texts: List of training texts
            labels: List of corresponding labels
            classifier_type: Type of classifier ('logistic', 'svm', 'random_forest')
            test_size: Fraction of data for testing
            random_state: Random seed

        Returns:
            Dict with training results
        """
        # Encode all texts
        print(f"Encoding {len(texts)} texts...")
        embeddings = self.encode(texts)

        # Encode labels to integers
        unique_labels = sorted(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}

        y = np.array([self.label_encoder[label] for label in labels])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, y, test_size=test_size, random_state=random_state
        )

        # Create classifier
        if classifier_type == "logistic":
            self.classifier = LogisticRegression(max_iter=1000, random_state=random_state)
        elif classifier_type == "svm":
            self.classifier = SVC(kernel='linear', probability=True, random_state=random_state)
        elif classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        # Train
        print(f"Training {classifier_type} classifier...")
        self.classifier.fit(X_train, y_train)

        # Evaluate
        train_acc = self.classifier.score(X_train, y_train)
        test_acc = self.classifier.score(X_test, y_test)

        # Cross-validation
        cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=5)

        # Predictions
        y_pred = self.classifier.predict(X_test)

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'true_labels': y_test,
            'num_classes': len(unique_labels),
            'classes': unique_labels
        }

    def predict(
        self,
        texts: List[str],
        return_probabilities: bool = False
    ) -> List[Dict]:
        """
        Predict labels for texts.

        Args:
            texts: List of texts to classify
            return_probabilities: Whether to return class probabilities

        Returns:
            List of dicts with predicted label and optionally probabilities
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train() first.")

        # Encode texts
        embeddings = self.encode(texts)

        # Predict
        pred_indices = self.classifier.predict(embeddings)

        results = []

        if return_probabilities and hasattr(self.classifier, 'predict_proba'):
            probs = self.classifier.predict_proba(embeddings)

            for idx, pred_idx in enumerate(pred_indices):
                probs_dict = {
                    self.label_decoder[i]: float(probs[idx, i])
                    for i in range(len(self.label_decoder))
                }

                results.append({
                    'predicted_label': self.label_decoder[pred_idx],
                    'probabilities': probs_dict,
                    'confidence': float(probs[idx, pred_idx])
                })
        else:
            for pred_idx in pred_indices:
                results.append({
                    'predicted_label': self.label_decoder[pred_idx]
                })

        return results

    def predict_single(self, text: str, return_probabilities: bool = False) -> Dict:
        """Predict label for a single text."""
        return self.predict([text], return_probabilities)[0]

    def get_feature_importance(self, top_k: int = 10) -> Dict:
        """
        Get feature importance (for classifiers that support it).

        Args:
            top_k: Number of top features to return

        Returns:
            Dict with feature importance scores
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained.")

        if hasattr(self.classifier, 'feature_importances_'):
            # Random Forest
            importance = self.classifier.feature_importances_
            top_indices = np.argsort(-importance)[:top_k]

            return {
                f'dimension_{i}': float(importance[i])
                for i in top_indices
            }
        elif hasattr(self.classifier, 'coef_'):
            # Logistic Regression or SVM (linear)
            importance = np.abs(self.classifier.coef_).mean(axis=0)
            top_indices = np.argsort(-importance)[:top_k]

            return {
                f'dimension_{i}': float(importance[i])
                for i in top_indices
            }
        else:
            return {}


def example_disease_classification():
    """Example: Classify medical conditions by body system."""
    print("\n" + "=" * 70)
    print("Example 1: Disease Classification by Body System")
    print("=" * 70)

    classifier = MedVectorsClassifier(model_name="abhinand/MedVectors-small-v0.1")

    # Training data
    texts = [
        # Cardiovascular
        "Myocardial infarction causes chest pain and elevated cardiac enzymes.",
        "Hypertension is defined as blood pressure above 130/80 mmHg.",
        "Atrial fibrillation leads to irregular heart rhythms.",
        "Heart failure presents with dyspnea and peripheral edema.",
        # Respiratory
        "Asthma is characterized by reversible airway obstruction.",
        "Pneumonia inflames the alveoli and causes fever.",
        "COPD includes chronic bronchitis and emphysema.",
        "Pulmonary embolism blocks pulmonary arteries.",
        # Gastrointestinal
        "Gastritis causes inflammation of the stomach lining.",
        "Crohn's disease is a type of inflammatory bowel disease.",
        "Hepatitis involves liver inflammation.",
        "Pancreatitis causes severe abdominal pain."
    ]

    labels = [
        "Cardiovascular", "Cardiovascular", "Cardiovascular", "Cardiovascular",
        "Respiratory", "Respiratory", "Respiratory", "Respiratory",
        "Gastrointestinal", "Gastrointestinal", "Gastrointestinal", "Gastrointestinal"
    ]

    # Train classifier
    results = classifier.train(texts, labels, classifier_type="logistic")

    print(f"\nTraining Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"CV Accuracy: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")

    # Predict on new samples
    new_texts = [
        "Patient has irregular heartbeat and palpitations.",
        "Chest X-ray shows infiltrates consistent with infection.",
        "Patient presents with severe abdominal pain and nausea."
    ]

    print("\nPredictions:")
    print("-" * 70)
    predictions = classifier.predict(new_texts, return_probabilities=True)
    for text, pred in zip(new_texts, predictions):
        print(f"\nText: {text}")
        print(f"Predicted: {pred['predicted_label']} (confidence: {pred['confidence']:.4f})")


def example_triage_classification():
    """Example: Classify cases by urgency level."""
    print("\n" + "=" * 70)
    print("Example 2: Triage Level Classification")
    print("=" * 70)

    classifier = MedVectorsClassifier(model_name="abhinand/MedVectors-base-v0.1")

    # Triage scenarios
    scenarios = [
        # Emergency
        "Patient with chest pain radiating to left arm, diaphoresis.",
        "Unresponsive patient with no palpable pulse.",
        "Severe shortness of breath with cyanosis.",
        # Urgent
        "High fever of 103°F with confusion.",
        "Severe abdominal pain with rigidity.",
        "Compound fracture with visible bone.",
        # Non-urgent
        "Mild headache for 2 days.",
        "Rash on forearm for 1 week.",
        "Occasional dry cough, otherwise asymptomatic.",
        # Self-care
        "Minor scrape on knee from fall.",
        "Mild sunburn on shoulders.",
        "Occasional indigestion after meals."
    ]

    triage_labels = [
        "Emergency", "Emergency", "Emergency",
        "Urgent", "Urgent", "Urgent",
        "Non-urgent", "Non-urgent", "Non-urgent",
        "Self-care", "Self-care", "Self-care"
    ]

    # Train classifier
    results = classifier.train(scenarios, triage_labels, classifier_type="random_forest")

    print(f"\nTraining Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")

    # Predict on new cases
    new_cases = [
        "Patient reports severe chest pain and difficulty breathing.",
        "Child has fever of 102°F and is lethargic.",
        "Adult reports minor scratch on finger."
    ]

    print("\nTriage Predictions:")
    print("-" * 70)
    predictions = classifier.predict(new_cases)
    for case, pred in zip(new_cases, predictions):
        print(f"\nCase: {case}")
        print(f"Triage Level: {pred['predicted_label']}")


def example_specialty_routing():
    """Example: Route patients to medical specialties."""
    print("\n" + "=" * 70)
    print("Example 3: Medical Specialty Routing")
    print("=" * 70)

    classifier = MedVectorsClassifier(model_name="abhinand/MedVectors-small-v0.1")

    # Patient complaints
    complaints = [
        # Cardiology
        "Patient reports chest pain and palpitations.",
        "Heart rate irregular with history of atrial fibrillation.",
        # Neurology
        "Sudden onset right-sided weakness and speech difficulty.",
        "Chronic migraines with visual aura.",
        # Gastroenterology
        "Chronic abdominal pain and diarrhea.",
        "Patient has difficulty swallowing solid foods.",
        # Dermatology
        "Rash spreading on arms with itching.",
        "Suspicious mole with irregular borders.",
        # Orthopedics
        "Knee pain worse with physical activity.",
        "Patient has back pain radiating to legs."
    ]

    specialties = [
        "Cardiology", "Cardiology",
        "Neurology", "Neurology",
        "Gastroenterology", "Gastroenterology",
        "Dermatology", "Dermatology",
        "Orthopedics", "Orthopedics"
    ]

    # Train classifier
    results = classifier.train(complaints, specialties, classifier_type="svm")

    print(f"\nTraining Accuracy: {results['train_accuracy']:.4f}")
    print(f"CV Accuracy: {results['cv_mean']:.4f}")

    # Predict on new patients
    new_patients = [
        "Patient reports severe headache with nausea.",
        "Swollen ankle after twist injury.",
        "Chronic constipation and bloating."
    ]

    print("\nSpecialty Recommendations:")
    print("-" * 70)
    predictions = classifier.predict(new_patients)
    for patient, pred in zip(new_patients, predictions):
        print(f"\nComplaint: {patient}")
        print(f"Refer to: {pred['predicted_label']}")


def example_symptom_diagnosis():
    """Example: Classify symptoms to potential conditions."""
    print("\n" + "=" * 70)
    print("Example 4: Symptom-Based Diagnosis")
    print("=" * 70)

    classifier = MedVectorsClassifier(model_name="abhinand/MedVectors-base-v0.1")

    # Symptom descriptions
    symptoms = [
        # Diabetes
        "Increased thirst and frequent urination.",
        "Unexplained weight loss and fatigue.",
        # Influenza
        "Fever, body aches, and sore throat.",
        "Runny nose, cough, and malaise.",
        # Migraine
        "Severe unilateral headache with nausea.",
        "Headache with visual aura and sensitivity to light.",
        # Pneumonia
        "High fever, cough with yellow sputum.",
        "Chest pain and difficulty breathing.",
        # Appendicitis
        "Abdominal pain starting near navel, moving to right lower quadrant.",
        "Pain worsens with movement and coughing."
    ]

    conditions = [
        "Diabetes", "Diabetes",
        "Influenza", "Influenza",
        "Migraine", "Migraine",
        "Pneumonia", "Pneumonia",
        "Appendicitis", "Appendicitis"
    ]

    # Train classifier
    results = classifier.train(symptoms, conditions, classifier_type="logistic")

    print(f"\nTraining Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")

    # Predict on new symptoms
    new_symptoms = [
        "Patient has excessive thirst and urinates frequently.",
        "Severe headache with light sensitivity.",
        "Right lower quadrant abdominal pain with fever."
    ]

    print("\nDiagnosis Predictions:")
    print("-" * 70)
    predictions = classifier.predict(new_symptoms, return_probabilities=True)
    for symptom, pred in zip(new_symptoms, predictions):
        print(f"\nSymptoms: {symptom}")
        print(f"Most likely: {pred['predicted_label']} (confidence: {pred['confidence']:.4f})")

        # Show top 2 alternatives
        sorted_probs = sorted(pred['probabilities'].items(), key=lambda x: x[1], reverse=True)
        print("Top 2 possibilities:")
        for condition, prob in sorted_probs[:2]:
            print(f"  {condition}: {prob:.4f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MedVectors Classification Examples")
    print("=" * 70)

    example_disease_classification()
    example_triage_classification()
    example_specialty_routing()
    example_symptom_diagnosis()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
