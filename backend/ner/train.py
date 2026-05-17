"""
SpaCy fine-tuning script for custom legal NER.

Usage:
    python train.py

Outputs a saved model to  ../models/legal_ner/
Requirements: pip install spacy[transformers]  (or just spacy for CPU)

F1 evaluation is printed at the end of training.
"""
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import json
import os

from data.training_data import TRAINING_DATA

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "legal_ner")
LABELS = ["LEGAL_DATE", "LEGAL_AMOUNT", "LEGAL_PARTY", "TERMINATION_CLAUSE"]
N_ITER = 30


def train():
    # Load blank model with English tokenizer
    nlp = spacy.blank("en")

    # Add NER pipe
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for label in LABELS:
        ner.add_label(label)

    # Build Example objects
    examples = []
    for text, annotations in TRAINING_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)

    # Split 80/20
    random.shuffle(examples)
    split = int(len(examples) * 0.8)
    train_ex = examples[:split]
    test_ex = examples[split:]

    # Training loop
    optimizer = nlp.begin_training()
    for i in range(N_ITER):
        random.shuffle(train_ex)
        losses = {}
        batches = minibatch(train_ex, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(batch, sgd=optimizer, drop=0.35, losses=losses)
        if (i + 1) % 5 == 0:
            print(f"[Iter {i+1:02d}] Loss: {losses.get('ner', 0):.4f}")

    # Evaluate
    scorer = nlp.evaluate(test_ex)
    print("\n── Evaluation ──────────────────────────────────")
    ents_f = scorer.get("ents_f", 0)
    ents_p = scorer.get("ents_p", 0)
    ents_r = scorer.get("ents_r", 0)
    print(f"  F1:        {ents_f:.4f}")
    print(f"  Precision: {ents_p:.4f}")
    print(f"  Recall:    {ents_r:.4f}")

    per_type = scorer.get("ents_per_type", {})
    for label, scores in per_type.items():
        print(f"  {label}: F1={scores.get('f', 0):.3f}")

    # Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nlp.to_disk(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")

    return scorer


if __name__ == "__main__":
    train()
