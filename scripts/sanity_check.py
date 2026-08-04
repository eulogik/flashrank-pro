#!/usr/bin/env python3
"""Quick sanity check for the reranker model."""
import time
import numpy as np
from sentence_transformers import CrossEncoder

def main():
    model_path = "eulogik/flashrank-pro-base"
    print(f"Loading model from {model_path}...")
    
    start = time.time()
    reranker = CrossEncoder(model_path, max_length=512, trust_remote_code=True)
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    
    test_cases = [
        {
            "query": "What is the capital of France?",
            "docs": [
                "Paris is the capital and most populous city of France.",
                "The Eiffel Tower is located in Paris, France.",
                "France is a country in Western Europe.",
                "Berlin is the capital of Germany.",
                "London is the capital of the United Kingdom.",
            ],
            "expected_top": 0,  # First doc should be highest
        },
        {
            "query": "How do I train a neural network?",
            "docs": [
                "Neural networks are trained using backpropagation.",
                "The weather is sunny today.",
                "Training requires a dataset and optimization algorithm.",
                "Python is a programming language.",
                "Deep learning uses neural networks with many layers.",
            ],
            "expected_top": 0,  # First doc should be highest
        },
        {
            "query": "What is machine learning?",
            "docs": [
                "Machine learning is a subset of artificial intelligence.",
                "The cat sat on the mat.",
                "ML algorithms learn patterns from data.",
                "Python is a programming language.",
                "Deep learning is a subset of machine learning.",
            ],
            "expected_top": 0,  # First doc should be highest
        },
    ]
    
    print("\nRunning sanity checks...")
    correct = 0
    total = 0
    
    for i, test in enumerate(test_cases):
        query = test["query"]
        docs = test["docs"]
        pairs = [[query, doc] for doc in docs]
        
        start = time.time()
        scores = reranker.predict(pairs)
        pred_time = time.time() - start
        
        top_idx = np.argmax(scores)
        expected = test["expected_top"]
        
        status = "✓" if top_idx == expected else "✗"
        if top_idx == expected:
            correct += 1
        total += 1
        
        print(f"\nTest {i+1}: {status}")
        print(f"  Query: {query}")
        print(f"  Top doc: {docs[top_idx]}")
        print(f"  Score: {scores[top_idx]:.4f}")
        print(f"  Time: {pred_time:.3f}s")
        print(f"  All scores: {[f'{s:.4f}' for s in scores]}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"Accuracy: {correct}/{total} ({accuracy*100:.1f}%)")
    print(f"{'='*50}")
    
    if accuracy >= 0.8:
        print("✓ Model is performing well!")
        return 0
    else:
        print("✗ Model performance needs improvement")
        return 1

if __name__ == "__main__":
    exit(main())
