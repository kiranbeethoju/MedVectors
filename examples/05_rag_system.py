"""
MedVectors RAG System Example
==============================

This example demonstrates building a Retrieval-Augmented Generation (RAG) system
using MedVectors for semantic search.

Use Cases:
- Medical question answering
- Clinical decision support
- Medical literature search
- Knowledge base retrieval
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Optional
import json


class MedVectorsRAG:
    """RAG system using MedVectors for retrieval."""

    def __init__(
        self,
        model_name: str = "kiranbeethoju/MedVectors-base-v0.1",
        device: str = None,
        max_length: int = 512
    ):
        """
        Initialize MedVectors RAG system.

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

        # Initialize knowledge base
        self.knowledge_base = []
        self.knowledge_embeddings = None

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

    def add_documents(self, documents: List[Dict]):
        """
        Add documents to the knowledge base.

        Args:
            documents: List of dicts with keys: id, text, title (optional), metadata (optional)
        """
        self.knowledge_base = documents
        texts = [doc['text'] for doc in documents]
        print(f"Encoding {len(texts)} documents...")
        self.knowledge_embeddings = self.encode(texts)
        print(f"✅ Knowledge base ready: {len(self.knowledge_base)} documents")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_metadata: Dict = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents from knowledge base.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            min_score: Minimum similarity score
            filter_metadata: Optional metadata filter

        Returns:
            List of retrieved documents with scores
        """
        if self.knowledge_embeddings is None:
            raise ValueError("Knowledge base is empty. Call add_documents() first.")

        # Encode query
        query_embedding = self.encode([query])[0]

        # Compute cosine similarity
        scores = np.dot(self.knowledge_embeddings, query_embedding)

        # Filter by metadata if specified
        valid_indices = list(range(len(self.knowledge_base)))
        if filter_metadata:
            valid_indices = [
                idx for idx in valid_indices
                if self._matches_filter(self.knowledge_base[idx], filter_metadata)
            ]

        # Get valid scores
        valid_scores = scores[valid_indices]

        # Get top-k results
        top_indices_local = np.argsort(-valid_scores)[:top_k]

        results = []
        for idx in top_indices_local:
            doc_idx = valid_indices[idx]
            if valid_scores[idx] >= min_score:
                result = self.knowledge_base[doc_idx].copy()
                result['score'] = float(valid_scores[idx])
                results.append(result)

        return results

    def _matches_filter(self, document: Dict, filter_metadata: Dict) -> bool:
        """Check if document matches metadata filter."""
        doc_metadata = document.get('metadata', {})

        for key, value in filter_metadata.items():
            if doc_metadata.get(key) != value:
                return False

        return True

    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Dict],
        llm_prompt: Optional[str] = None
    ) -> str:
        """
        Generate answer using retrieved context (requires LLM integration).

        Args:
            query: User question
            retrieved_docs: Retrieved documents
            llm_prompt: Custom prompt template

        Returns:
            Generated answer (placeholder - requires actual LLM)
        """
        # This is a placeholder for LLM integration
        # In practice, you would integrate with OpenAI, Anthropic, etc.

        context = "\n\n".join([
            f"Document {i+1}: {doc['text']}"
            for i, doc in enumerate(retrieved_docs)
        ])

        if llm_prompt is None:
            llm_prompt = """Based on the following medical documents, answer the question.

Documents:
{context}

Question: {question}

Answer:"""

        answer = llm_prompt.format(context=context, question=query)

        # NOTE: This returns the prompt, not an actual answer
        # In production, send this to an LLM and return its response
        return f"[LLM Integration Required]\n\nContext retrieved from {len(retrieved_docs)} documents:\n{context}"


def example_medical_qa():
    """Example: Medical question answering system."""
    print("\n" + "=" * 70)
    print("Example 1: Medical Q&A System")
    print("=" * 70)

    rag = MedVectorsRAG(model_name="kiranbeethoju/MedVectors-base-v0.1")

    # Build knowledge base
    documents = [
        {
            'id': 1,
            'title': 'Type 2 Diabetes',
            'text': 'Type 2 diabetes is a chronic condition where the body cannot use insulin properly, leading to high blood sugar. Risk factors include obesity, physical inactivity, family history, and age over 45. Treatment involves lifestyle changes and medications like metformin, GLP-1 agonists, or insulin.',
            'metadata': {'category': 'Endocrine', 'severity': 'Chronic'}
        },
        {
            'id': 2,
            'title': 'Myocardial Infarction',
            'text': 'Myocardial infarction (heart attack) occurs when blood flow to the heart is blocked, usually by a blood clot. Symptoms include chest pain, shortness of breath, sweating, nausea, and pain radiating to the left arm or jaw. Immediate treatment includes aspirin, nitroglycerin, and emergency revascularization procedures.',
            'metadata': {'category': 'Cardiovascular', 'severity': 'Emergency'}
        },
        {
            'id': 3,
            'title': 'Asthma',
            'text': 'Asthma is a chronic inflammatory disease of the airways characterized by variable and recurring symptoms, reversible airflow obstruction, and bronchospasm. Triggers include allergens, exercise, cold air, and stress. Treatment involves inhaled corticosteroids and bronchodilators.',
            'metadata': {'category': 'Respiratory', 'severity': 'Chronic'}
        },
        {
            'id': 4,
            'title': 'Pneumonia',
            'text': 'Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus. Symptoms include fever, cough, difficulty breathing, and chest pain. Treatment depends on the cause: antibiotics for bacterial pneumonia, antivirals for viral, and supportive care.',
            'metadata': {'category': 'Respiratory', 'severity': 'Acute'}
        },
        {
            'id': 5,
            'title': 'Hypertension',
            'text': 'Hypertension (high blood pressure) is defined as blood pressure of 130/80 mmHg or higher. It is a major risk factor for heart disease and stroke. Lifestyle modifications include reducing sodium intake, exercising regularly, maintaining healthy weight, and limiting alcohol. Medications include ACE inhibitors, beta-blockers, and diuretics.',
            'metadata': {'category': 'Cardiovascular', 'severity': 'Chronic'}
        }
    ]

    rag.add_documents(documents)

    # Answer questions
    questions = [
        "What are the risk factors for type 2 diabetes?",
        "What are the symptoms of a heart attack?",
        "How is asthma treated?",
        "What causes pneumonia?"
    ]

    print("\nAnswering Questions:")
    print("-" * 70)

    for question in questions:
        print(f"\n❓ Question: {question}")

        results = rag.retrieve(question, top_k=2)

        if results:
            print(f"✅ Most relevant document (Score: {results[0]['score']:.4f}):")
            print(f"   {results[0]['title']}")
            print(f"   {results[0]['text'][:200]}...")
        else:
            print("❌ No relevant documents found.")


def example_clinical_decision_support():
    """Example: Clinical decision support system."""
    print("\n" + "=" * 70)
    print("Example 2: Clinical Decision Support")
    print("=" * 70)

    rag = MedVectorsRAG(model_name="kiranbeethoju/MedVectors-base-v0.1")

    # Clinical guidelines knowledge base
    guidelines = [
        {
            'id': 1,
            'title': 'Acute MI Management',
            'text': 'For acute myocardial infarction, administer aspirin 325mg chewed immediately. If no contraindications, give nitroglycerin 0.4mg sublingual. Monitor blood pressure. For STEMI, arrange immediate cardiac catheterization for primary PCI within 90 minutes.',
            'metadata': {'category': 'Cardiovascular', 'type': 'Emergency'}
        },
        {
            'id': 2,
            'title': 'Anaphylaxis Treatment',
            'text': 'For anaphylaxis, administer epinephrine 0.3-0.5mg IM immediately in the mid-anterolateral thigh. May repeat every 5-15 minutes as needed. Provide oxygen, establish IV access, give diphenhydramine 25-50mg IV or IM, and methylprednisolone 125mg IV.',
            'metadata': {'category': 'Allergy', 'type': 'Emergency'}
        },
        {
            'id': 3,
            'title': 'DKA Management',
            'text': 'For diabetic ketoacidosis, administer IV isotonic saline at 500-1000mL initially. Start regular insulin at 0.1 units/kg/hour after potassium is >3.3 mEq/L. Check blood glucose every hour. When glucose reaches 200mg/dL, add D5W to IV fluids. Continue insulin until anion gap closes.',
            'metadata': {'category': 'Endocrine', 'type': 'Emergency'}
        },
        {
            'id': 4,
            'title': 'Sepsis Management',
            'text': 'For sepsis, initiate broad-spectrum antibiotics within 1 hour after obtaining cultures. Administer 30mL/kg crystalloid IV bolus. Consider vasopressors (norepinephrine first-line) if hypotensive after fluids. Maintain MAP >65 mmHg. Source control within 6 hours.',
            'metadata': {'category': 'Infectious', 'type': 'Emergency'}
        }
    ]

    rag.add_documents(guidelines)

    # Clinical scenarios
    scenarios = [
        "Patient with severe allergic reaction, swelling, and difficulty breathing",
        "Patient with blood sugar 450, positive ketones, and metabolic acidosis",
        "Patient with fever 103°F, low blood pressure, and suspected infection"
    ]

    print("\nClinical Decision Support:")
    print("-" * 70)

    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario}")

        results = rag.retrieve(scenario, top_k=1, min_score=0.3)

        if results:
            print(f"📖 Relevant Guideline (Score: {results[0]['score']:.4f}):")
            print(f"   {results[0]['title']}")
            print(f"   {results[0]['text'][:300]}...")
        else:
            print("❌ No relevant guidelines found.")


def example_medical_literature_search():
    """Example: Search medical literature."""
    print("\n" + "=" * 70)
    print("Example 3: Medical Literature Search")
    print("=" * 70)

    rag = MedVectorsRAG(model_name="kiranbeethoju/MedVectors-base-v0.1")

    # Abstracts from medical literature
    abstracts = [
        {
            'id': 1,
            'title': 'GLP-1 Receptor Agonists in Type 2 Diabetes',
            'text': 'This meta-analysis evaluated the efficacy of GLP-1 receptor agonists in type 2 diabetes. Results showed significant reductions in HbA1c (-0.8% to -1.5%), weight loss (-2.0 to -5.4 kg), and reduced cardiovascular events compared to placebo. Side effects include gastrointestinal symptoms and rare pancreatitis.',
            'metadata': {'category': 'Diabetes', 'type': 'Meta-analysis'}
        },
        {
            'id': 2,
            'title': 'Direct Oral Anticoagulants vs Warfarin in AF',
            'text': 'This systematic review compared DOACs to warfarin for stroke prevention in atrial fibrillation. DOACs showed similar efficacy for stroke prevention but lower rates of major bleeding. All DOACs (apixaban, rivaroxaban, dabigatran, edoxaban) were non-inferior to warfarin with favorable safety profiles.',
            'metadata': {'category': 'Cardiology', 'type': 'Systematic Review'}
        },
        {
            'id': 3,
            'title': 'SGLT2 Inhibitors in Heart Failure',
            'text': 'Recent trials demonstrate that SGLT2 inhibitors reduce hospitalization for heart failure and cardiovascular death in patients with HFrEF and HFpEF. Benefits appear independent of diabetes status and begin within weeks of initiation. Recommended as part of guideline-directed medical therapy.',
            'metadata': {'category': 'Cardiology', 'type': 'Clinical Trial'}
        },
        {
            'id': 4,
            'title': 'Immunotherapy in Non-Small Cell Lung Cancer',
            'text': 'Checkpoint inhibitors (pembrolizumab, nivolumab) have revolutionized NSCLC treatment. For PD-L1 expression >50%, pembrolizumab monotherapy shows superior OS compared to chemotherapy. Combination immunotherapy+chemotherapy benefits patients regardless of PD-L1 status but increases toxicity.',
            'metadata': {'category': 'Oncology', 'type': 'Clinical Trial'}
        }
    ]

    rag.add_documents(abstracts)

    # Search queries
    queries = [
        "What are the benefits of SGLT2 inhibitors?",
        "GLP-1 agonists for diabetes",
        "Anticoagulation for atrial fibrillation",
        "First-line treatment for lung cancer"
    ]

    print("\nLiterature Search:")
    print("-" * 70)

    for query in queries:
        print(f"\n🔍 Query: {query}")

        results = rag.retrieve(query, top_k=1, min_score=0.4)

        if results:
            print(f"📄 Best Match (Score: {results[0]['score']:.4f}):")
            print(f"   {results[0]['title']}")
            print(f"   {results[0]['text'][:200]}...")
        else:
            print("❌ No matches found.")


def example_filtered_retrieval():
    """Example: Retrieval with metadata filtering."""
    print("\n" + "=" * 70)
    print("Example 4: Filtered Retrieval")
    print("=" * 70)

    rag = MedVectorsRAG(model_name="kiranbeethoju/MedVectors-small-v0.1")

    # Knowledge base with metadata
    documents = [
        {
            'id': 1,
            'text': 'Cardiovascular conditions: hypertension, myocardial infarction, heart failure, arrhythmias.',
            'metadata': {'category': 'Cardiovascular', 'level': 'General'}
        },
        {
            'id': 2,
            'text': 'Respiratory conditions: asthma, COPD, pneumonia, pulmonary embolism.',
            'metadata': {'category': 'Respiratory', 'level': 'General'}
        },
        {
            'id': 3,
            'text': 'Endocrine disorders: diabetes, thyroid disorders, adrenal insufficiency.',
            'metadata': {'category': 'Endocrine', 'level': 'General'}
        },
        {
            'id': 4,
            'text': 'Acute MI management: aspirin, nitroglycerin, heparin, and timely PCI.',
            'metadata': {'category': 'Cardiovascular', 'level': 'Emergency'}
        },
        {
            'id': 5,
            'text': 'Diabetic ketoacidosis: insulin therapy, fluid resuscitation, electrolyte monitoring.',
            'metadata': {'category': 'Endocrine', 'level': 'Emergency'}
        }
    ]

    rag.add_documents(documents)

    # Search with filter
    query = "Emergency treatment needed"
    print(f"\nQuery: {query}")

    # Search only emergency protocols
    print("\n🔍 Retrieving emergency protocols only:")
    results = rag.retrieve(
        query,
        top_k=3,
        filter_metadata={'level': 'Emergency'}
    )

    for result in results:
        print(f"  {result['metadata']['category']}: {result['text'][:60]}...")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MedVectors RAG System Examples")
    print("=" * 70)

    example_medical_qa()
    example_clinical_decision_support()
    example_medical_literature_search()
    example_filtered_retrieval()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
