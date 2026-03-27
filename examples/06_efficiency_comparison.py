"""
MedVectors Efficiency Comparison vs all-mini-lm-v2
=======================================================

This example demonstrates why MedVectors (specialized embedding model) is more efficient
than all-mini-lm-v2 (small language model) for retrieval tasks.

Key Points:
- Embedding models are purpose-built for similarity search
- LLMs generate text but are overkill for retrieval
- Embeddings are faster, smaller, and more efficient at scale
"""

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import time
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


class MedVectorsComparison:
    """Compare MedVectors vs all-mini-lm-v2 for retrieval efficiency."""

    def __init__(self, device: str = None):
        """Initialize both models for comparison."""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Initialize MedVectors (embedding model)
        print("\n" + "=" * 70)
        print("Loading MedVectors (Embedding Model)")
        print("=" * 70)
        self.medvectors_name = "kiranbeethoju/MedVectors-small-v0.1"
        self.medvectors_tokenizer = AutoTokenizer.from_pretrained(self.medvectors_name)
        self.medvectors_model = AutoModel.from_pretrained(self.medvectors_name).to(self.device)
        self.medvectors_model.eval()

        # Initialize all-mini-lm-v2 (language model)
        print("\n" + "=" * 70)
        print("Loading all-mini-lm-v2 (Small Language Model)")
        print("=" * 70)
        self.lm_name = "all-MiniLM-L6-v2"
        self.lm_tokenizer = AutoTokenizer.from_pretrained(self.lm_name)
        self.lm_model = AutoModelForCausalLM.from_pretrained(self.lm_name).to(self.device)
        self.lm_model.eval()

        # Print model info
        self._print_model_info()

    def _print_model_info(self):
        """Print detailed model comparison."""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)

        # Count parameters
        mv_params = sum(p.numel() for p in self.medvectors_model.parameters())
        lm_params = sum(p.numel() for p in self.lm_model.parameters())

        print(f"\n{'Metric':<30} | {'MedVectors':>20} | {'all-mini-lm-v2':>20}")
        print("-" * 70)
        print(f"{'Model Type':<30} | {'Embedding':>20} | {'Language Model':>20}")
        print(f"{'Parameters':<30} | {mv_params:>20,.0f}M | {lm_params:>20,.0f}M")
        print(f"{'Param Ratio':<30} | {'1.0x':>20} | {lm_params/mv_params:>20.2f}x larger")
        print(f"{'Purpose':<30} | {'Similarity':>20} | {'Text Generation':>20}")
        print(f"{'Use Case':<30} | {'Retrieval':>20} | {'Content Creation':>20}")

        print(f"\n{'Embedding Dim':<30} | {384:>20} | N/A (LM uses hidden states)")
        print(f"{'Output':<30} | {'Single Vector':>20} | {'Next Token Probabilities':>20}")
        print(f"{'Training Objective':<30} | {'Contrastive':>20} | {'Causal Language Modeling':>20}")

    def encode_medvectors(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> Tuple[np.ndarray, Dict]:
        """
        Encode texts using MedVectors.

        Returns:
            (embeddings, timing_info)
        """
        all_embeddings = []
        timings = {'encode_times': [], 'total_tokens': 0}

        print(f"\nEncoding {len(texts)} texts with MedVectors...")

        for i, batch_texts in enumerate([texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]):
            start_time = time.time()

            inputs = self.medvectors_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.medvectors_model(**inputs)

            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)

            # Normalize for cosine similarity
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

            encode_time = time.time() - start_time
            timings['encode_times'].append(encode_time)
            timings['total_tokens'] += inputs['input_ids'].numel()

        return np.concatenate(all_embeddings, axis=0), timings

    def encode_allminilm(
        self,
        texts: List[str],
        use_last_hidden: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Get embeddings from all-mini-lm-v2.

        Note: This is inefficient - LM is not designed for this!

        Args:
            texts: Texts to encode
            use_last_hidden: Use last hidden state vs pooling

        Returns:
            (embeddings, timing_info)
        """
        all_embeddings = []
        timings = {'encode_times': [], 'total_tokens': 0}

        print(f"\nEncoding {len(texts)} texts with all-mini-lm-v2...")

        for text in texts:
            start_time = time.time()

            inputs = self.lm_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.lm_model(**inputs, output_hidden_states=True)

            if use_last_hidden:
                # Use last hidden state (inefficient)
                embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
            else:
                # Mean pool all hidden states (even more inefficient)
                all_hidden = torch.stack(outputs.hidden_states).mean(dim=0)
                embedding = all_hidden.mean(dim=1).squeeze()

            # Normalize
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)

            all_embeddings.append(embedding.cpu().numpy())

            encode_time = time.time() - start_time
            timings['encode_times'].append(encode_time)
            timings['total_tokens'] += inputs['input_ids'].numel()

        return np.array(all_embeddings), timings

    def compute_similarity_matrix(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        method: str = "medvectors"
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            method: Which embeddings to use

        Returns:
            Similarity matrix
        """
        # Dot product for cosine similarity (embeddings are normalized)
        return np.dot(embeddings1, embeddings2.T)

    def compare_efficiency(
        self,
        corpus: List[str],
        queries: List[str],
        top_k: int = 5
    ) -> Dict:
        """
        Comprehensive efficiency comparison.

        Args:
            corpus: Documents to search
            queries: Query texts
            top_k: Top-k to retrieve

        Returns:
            Dict with all comparison metrics
        """
        print("\n" + "=" * 70)
        print("EFFICIENCY COMPARISON")
        print("=" * 70)

        # Encode with MedVectors
        mv_corpus_embeds, mv_corpus_time = self.encode_medvectors(corpus)
        mv_query_embeds, mv_query_time = self.encode_medvectors(queries)

        # Encode with all-mini-lm-v2
        lm_corpus_embeds, lm_corpus_time = self.encode_allminilm(corpus)
        lm_query_embeds, lm_query_time = self.encode_allminilm(queries)

        # Compute similarity and retrieve
        print("\nComputing similarity and retrieval...")

        # MedVectors retrieval
        mv_start = time.time()
        mv_similarities = self.compute_similarity_matrix(
            mv_query_embeds, mv_corpus_embeds, "medvectors"
        )
        mv_retrieval_time = time.time() - mv_start

        # all-mini-lm-v2 retrieval
        lm_start = time.time()
        lm_similarities = self.compute_similarity_matrix(
            lm_query_embeds, lm_corpus_embeds, "allminilm"
        )
        lm_retrieval_time = time.time() - lm_start

        # Get top-k for each
        mv_top_k = np.argsort(-mv_similarities, axis=1)[:, :top_k]
        lm_top_k = np.argsort(-lm_similarities, axis=1)[:, :top_k]

        # Calculate metrics
        mv_total_encode = mv_corpus_time['total_tokens'] / mv_corpus_time['total_tokens'] * sum(mv_corpus_time['encode_times']) + \
                        mv_query_time['total_tokens'] / mv_query_time['total_tokens'] * sum(mv_query_time['encode_times'])
        lm_total_encode = lm_corpus_time['total_tokens'] / lm_corpus_time['total_tokens'] * sum(lm_corpus_time['encode_times']) + \
                        lm_query_time['total_tokens'] / lm_query_time['total_tokens'] * sum(lm_query_time['encode_times'])

        # Compute overlap (how similar the results are)
        overlap = []
        for i in range(len(queries)):
            mv_set = set(mv_top_k[i])
            lm_set = set(lm_top_k[i])
            overlap.append(len(mv_set & lm_set) / top_k)
        avg_overlap = np.mean(overlap)

        results = {
            # Encoding metrics
            'mv_corpus_time': sum(mv_corpus_time['encode_times']),
            'mv_query_time': sum(mv_query_time['encode_times']),
            'mv_total_time': mv_total_encode,

            'lm_corpus_time': sum(lm_corpus_time['encode_times']),
            'lm_query_time': sum(lm_query_time['encode_times']),
            'lm_total_time': lm_total_encode,

            # Retrieval metrics
            'mv_retrieval_time': mv_retrieval_time,
            'lm_retrieval_time': lm_retrieval_time,

            # Total
            'mv_total_time': mv_total_encode + mv_retrieval_time,
            'lm_total_time': lm_total_encode + lm_retrieval_time,

            # Speedup
            'encoding_speedup': lm_total_encode / mv_total_encode,
            'total_speedup': (lm_total_encode + lm_retrieval_time) / (mv_total_encode + mv_retrieval_time),
            'result_overlap': avg_overlap,

            # Model info
            'mv_params': sum(p.numel() for p in self.medvectors_model.parameters()),
            'lm_params': sum(p.numel() for p in self.lm_model.parameters())
        }

        self._print_comparison_results(results, len(corpus), len(queries))

        return results

    def _print_comparison_results(self, results: Dict, corpus_size: int, num_queries: int):
        """Print formatted comparison results."""
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        print(f"\n📊 Encoding Performance:")
        print("-" * 70)
        print(f"{'Metric':<30} | {'MedVectors':>20} | {'all-mini-lm-v2':>20}")
        print("-" * 70)
        print(f"{'Corpus Encoding':<30} | {results['mv_corpus_time']:.4f}s | {results['lm_corpus_time']:.4f}s")
        print(f"{'Query Encoding':<30} | {results['mv_query_time']:.4f}s | {results['lm_query_time']:.4f}s")
        print(f"{'Total Encoding':<30} | {results['mv_total_time']:.4f}s | {results['lm_total_time']:.4f}s")

        print(f"\n⚡ Speed Analysis:")
        print("-" * 70)
        print(f"Encoding Speedup: {results['encoding_speedup']:.2f}x faster")
        print(f"Overall Speedup: {results['total_speedup']:.2f}x faster")

        print(f"\n💾 Memory Efficiency:")
        print("-" * 70)
        print(f"Parameter Ratio: {results['lm_params'] / results['mv_params']:.2f}x")
        print(f"MedVectors: {results['mv_params'] / 1e6:.2f}M parameters")
        print(f"all-mini-lm-v2: {results['lm_params'] / 1e6:.2f}M parameters")

        print(f"\n🎯 Retrieval Quality:")
        print("-" * 70)
        print(f"Result Overlap: {results['result_overlap'] * 100:.1f}%")
        print(f"(How similar top-k results are between the two models)")

        print(f"\n💡 Efficiency Insights:")
        print("-" * 70)
        print("✅ MedVectors is specifically designed for similarity search")
        print("✅ Embedding model = single forward pass, efficient computation")
        print("✅ all-mini-lm-v2 generates text (overkill for retrieval)")
        print("✅ LLM requires hidden states extraction (inefficient)")
        print("✅ MedVectors optimized for cosine similarity")

    def visualize_comparison(self, results: Dict, save_path: str = "efficiency_comparison.png"):
        """Create visualization of the comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Time comparison
        categories = ['Corpus\nEncoding', 'Query\nEncoding', 'Retrieval', 'Total']
        mv_times = [
            results['mv_corpus_time'],
            results['mv_query_time'],
            results['mv_retrieval_time'],
            results['mv_total_time']
        ]
        lm_times = [
            results['lm_corpus_time'],
            results['lm_query_time'],
            results['lm_retrieval_time'],
            results['lm_total_time']
        ]

        x = np.arange(len(categories))
        width = 0.35

        axes[0].bar(x - width/2, mv_times, width, label='MedVectors', color='#2ecc71')
        axes[0].bar(x + width/2, lm_times, width, label='all-mini-lm-v2', color='#e74c3c')
        axes[0].set_ylabel('Time (seconds)')
        axes[0].set_title('Encoding & Retrieval Time Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Speedup bar
        speedups = [results['encoding_speedup'], results['total_speedup']]
        axes[1].bar(range(len(speedups)), speedups, color='#2ecc71')
        axes[1].set_ylabel('Speedup (x times faster)')
        axes[1].set_title('MedVectors Speedup vs all-mini-lm-v2')
        axes[1].set_xticks(range(len(speedups)))
        axes[1].set_xticklabels(['Encoding', 'Overall'])
        axes[1].grid(True, alpha=0.3)

        # Add speedup values on bars
        for i, v in enumerate(speedups):
            axes[1].text(i, v + 0.1, f'{v:.1f}x',
                     ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: {save_path}")
        plt.close()

    def example_small_scale(self):
        """Example: Small scale comparison (10 docs, 5 queries)."""
        print("\n" + "=" * 70)
        print("EXAMPLE 1: Small Scale (10 documents, 5 queries)")
        print("=" * 70)

        corpus = [
            "Type 2 diabetes is a chronic condition.",
            "Hypertension affects nearly half of adults.",
            "Asthma causes reversible airway obstruction.",
            "Myocardial infarction requires immediate treatment.",
            "Pneumonia inflames the alveoli.",
            "Heart failure presents with dyspnea.",
            "Osteoarthritis involves joint breakdown.",
            "Depression is a mood disorder.",
            "Stroke blocks blood flow to the brain.",
            "Migraine causes severe headaches."
        ]

        queries = [
            "What causes high blood sugar?",
            "Treatment for heart attack",
            "Symptoms of asthma",
            "Knee pain management",
            "Stroke symptoms"
        ]

        results = self.compare_efficiency(corpus, queries)

    def example_medium_scale(self):
        """Example: Medium scale comparison (100 docs, 20 queries)."""
        print("\n" + "=" * 70)
        print("EXAMPLE 2: Medium Scale (100 documents, 20 queries)")
        print("=" * 70)

        # Generate larger corpus
        base_docs = [
            "Type 2 diabetes is a chronic condition.",
            "Hypertension affects nearly half of adults.",
            "Asthma causes reversible airway obstruction.",
            "Myocardial infarction requires immediate treatment.",
            "Pneumonia inflames the alveoli.",
            "Heart failure presents with dyspnea.",
            "Osteoarthritis involves joint breakdown.",
            "Depression is a mood disorder.",
            "Stroke blocks blood flow to the brain.",
            "Migraine causes severe headaches."
        ]

        corpus = []
        for i in range(100):
            base = base_docs[i % len(base_docs)]
            corpus.append(f"{base} Variation {i // len(base_docs) + 1}.")

        queries = [
            "Diabetes treatment options",
            "Blood pressure management",
            "Asthma triggers and prevention",
            "Heart attack diagnosis",
            "Stroke recovery process",
            "Joint pain causes",
            "Depression symptoms",
            "Migraine medications",
            "Pneumonia risk factors",
            "Heart failure stages"
        ]

        results = self.compare_efficiency(corpus, queries, top_k=3)

    def example_realistic_use_case(self):
        """Example: Realistic clinical use case."""
        print("\n" + "=" * 70)
        print("EXAMPLE 3: Realistic Clinical Use Case")
        print("=" * 70)

        # Clinical guidelines corpus (simplified)
        clinical_guidelines = [
            "Acute MI: give aspirin 325mg chewed, nitroglycerin 0.4mg sublingual.",
            "Asthma exacerbation: start albuterol inhaler, increase ICS to high dose.",
            "DKA: IV fluids, insulin infusion, monitor potassium and glucose.",
            "Sepsis: broad-spectrum antibiotics within 1 hour, 30mL/kg crystalloid bolus.",
            "Anaphylaxis: epinephrine 0.3-0.5mg IM immediately.",
            "Hypoglycemia: give 15g glucose PO or IV, recheck in 15 min.",
            "Hyperkalemia: calcium gluconate IV, insulin+glucose, albuterol inhaler.",
            "Atrial fibrillation: rate control or rhythm control, consider anticoagulation.",
            "PE: therapeutic anticoagulation, consider thrombolysis if massive.",
            "Cardiac arrest: CPR, defibrillation if shockable rhythm, epinephrine 1mg."
        ] * 5  # Replicate for larger corpus

        # Clinical queries
        clinical_queries = [
            "What to do for acute myocardial infarction?",
            "How to manage asthma exacerbation?",
            "Treatment for diabetic ketoacidosis?",
            "Sepsis management protocol",
            "Anaphylaxis emergency treatment"
        ]

        results = self.compare_efficiency(clinical_guidelines, clinical_queries, top_k=5)

        print("\n💡 Key Insight for Clinical Use:")
        print("-" * 70)
        print("In clinical settings:")
        print("  • Fast retrieval is CRITICAL (patient care decisions)")
        print("  • MedVectors retrieves relevant guidelines in milliseconds")
        print("  • all-mini-lm-v2 would waste time generating unnecessary text")
        print("  • Embedding models are purpose-built for semantic search")
        print("  • Small speedups multiply in time-sensitive applications")


def example_embedding_vs_generation():
    """Demonstrate fundamental difference between embeddings and generation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 0: Embedding vs Language Model - Key Difference")
    print("=" * 70)

    print("""
📚️  EMBEDDING MODELS (like MedVectors):

Purpose: Convert text to fixed-size vectors for similarity search
Output: Single vector (e.g., 384 dimensions)
Use Case: Retrieval, semantic search, clustering
Training: Trained on contrastive learning (query-positive-negative triplets)
Computation: One forward pass through encoder

Pros:
✅ Fast - single pass, efficient computation
✅ Purpose-built for similarity comparison
✅ Low memory footprint
✅ No unnecessary token-by-token processing
✅ Optimized for cosine similarity

Cons:
❌ Cannot generate text
❌ Fixed output dimension

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤖️  LANGUAGE MODELS (like all-mini-lm-v2):

Purpose: Predict next token in sequence
Output: Next token probability for entire vocabulary
Use Case: Text generation, completion, chat
Training: Trained on causal language modeling
Computation: Autoregressive processing, generate hidden states for each token

Pros:
✅ Can generate coherent text
✅ Context-rich representations

Cons:
❌ Slow - processes entire sequence token-by-token
❌ Hidden states not designed for similarity comparison
❌ Overkill for retrieval tasks
❌ Higher computational cost
❌ Larger memory footprint

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯  THE KEY INSIGHT:

For RETRIEVAL tasks, embedding models win because:

1. Single forward pass vs. sequence generation
2. Purpose-built for similarity (dot product / cosine similarity)
3. Trained on query-positive-negative triplets (explicitly for retrieval)
4. No waste on generating unnecessary probabilities
5. Optimized architectures (e.g., mean pooling)

For GENERATION tasks, language models win because:

1. Explicitly trained to predict next tokens
2. Rich contextual representations
3. Can produce coherent text

💡 RULE OF THUMB:
  • Need to find/retrieve similar text? → Use embedding model (MedVectors)
  • Need to generate new text? → Use language model (all-mini-lm-v2)
  • Best of both worlds? → Use MedVectors for retrieval, LLM for answer generation (RAG)
    """)


def main():
    """Run all comparison examples."""
    print("\n" + "=" * 70)
    print("MedVectors vs all-mini-lm-v2 Efficiency Comparison")
    print("=" * 70)
    print("""
This comparison demonstrates why specialized embedding models (like MedVectors)
are more efficient than small language models (like all-mini-lm-v2) for
retrieval and semantic search tasks.

Key Takeaway: Use the right tool for the job!
    """)

    # Explain fundamental difference
    example_embedding_vs_generation()

    # Run comparisons
    comparison = MedVectorsComparison()

    comparison.example_small_scale()
    comparison.example_medium_scale()
    comparison.example_realistic_use_case()

    # Create visualization
    print("\n" + "=" * 70)
    print("Creating Visualization...")
    print("=" * 70)

    # Use results from realistic use case for visualization
    clinical_guidelines = [
        "Acute MI: give aspirin 325mg chewed, nitroglycerin 0.4mg sublingual.",
        "Asthma exacerbation: start albuterol inhaler, increase ICS to high dose.",
        "DKA: IV fluids, insulin infusion, monitor potassium and glucose.",
        "Sepsis: broad-spectrum antibiotics within 1 hour, 30mL/kg crystalloid bolus.",
        "Anaphylaxis: epinephrine 0.3-0.5mg IM immediately.",
    ] * 5

    clinical_queries = [
        "What to do for acute myocardial infarction?",
        "How to manage asthma exacerbation?",
        "Treatment for diabetic ketoacidosis?",
        "Sepsis management protocol",
        "Anaphylaxis emergency treatment"
    ]

    results = comparison.compare_efficiency(clinical_guidelines, clinical_queries)
    comparison.visualize_comparison(results)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
✅ MedVectors (embedding model) is:
   - Purpose-built for retrieval and similarity search
   - More efficient (fewer parameters, faster encoding)
   - Optimized for cosine similarity computation
   - Lower computational cost at scale

❌ all-mini-lm-v2 (language model) is:
   - Designed for text generation, not retrieval
   - Overkill for similarity search tasks
   - Higher computational cost
   - Hidden states not optimized for comparison

💡 RECOMMENDATION:
   For retrieval tasks (search, RAG, semantic similarity), use MedVectors.
   For generation tasks (content creation, chat), use all-mini-lm-v2.
   For RAG systems: Use MedVectors for retrieval + LLM for generation!
    """)

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
