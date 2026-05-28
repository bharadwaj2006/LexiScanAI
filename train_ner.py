"""
=============================================================
LEXISCAN AUTO — STEP 2: NER MODEL TRAINING (spaCy)
=============================================================
PURPOSE:
    Train a Named Entity Recognition model that recognises
    four custom legal entities inside contract text:
        DATE              — "March 15, 2024", "15/03/2024"
        PARTY             — "Meridian Capital LLC", "John Smith"
        DOLLAR_AMOUNT     — "$2,500,000", "USD 150,000"
        TERMINATION_CLAUSE— "Either party may terminate with 90 days notice"

HOW IT WORKS:
    1. Load training data (IOB2 annotated, from Doccano export).
    2. Build a spaCy NER pipeline on top of a blank English model.
    3. Train for N iterations with dropout and mini-batching.
    4. Evaluate with Precision, Recall, and F1-Score.
    5. Save the trained model to disk for use in Step 3 and Step 4.

LIBRARIES NEEDED:
    pip install spacy
    python -m spacy download en_core_web_sm

ANNOTATION FORMAT (what Doccano exports → what we convert to):
    TRAINING_DATA = [
        ("This Agreement is entered on March 15 2024 between Meridian Capital LLC ...",
         {"entities": [(38, 51, "DATE"), (60, 80, "PARTY")]}
        ),
        ...
    ]
    Each entity is a tuple: (start_char_index, end_char_index, LABEL)
=============================================================
"""

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random
import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SAMPLE TRAINING DATA
# ─────────────────────────────────────────────
# In a real project, this comes from your Doccano annotation export.
# Each entry: (text, {"entities": [(start, end, LABEL), ...]})
# start = character index where entity starts
# end   = character index where entity ends (exclusive)

SAMPLE_TRAINING_DATA = [
    (
        "This Agreement is entered into on March 15, 2024 between Meridian Capital LLC and Vantage Holdings Inc.",
        {"entities": [
            (34, 48, "DATE"),         # "March 15, 2024"
            (57, 78, "PARTY"),        # "Meridian Capital LLC"
            (83, 104, "PARTY"),       # "Vantage Holdings Inc."
        ]}
    ),
    (
        "The total contract value is $2,500,000.00 to be paid by January 31, 2025.",
        {"entities": [
            (27, 41, "DOLLAR_AMOUNT"),  # "$2,500,000.00"
            (53, 70, "DATE"),           # "January 31, 2025"
        ]}
    ),
    (
        "Either party may terminate this Agreement with 90 days written notice to the other party.",
        {"entities": [
            (0, 87, "TERMINATION_CLAUSE"),  # entire clause
        ]}
    ),
    (
        "ABC Legal Partners LLP agrees to pay $150,000 upon signing on 01/06/2024.",
        {"entities": [
            (0, 21, "PARTY"),          # "ABC Legal Partners LLP"
            (37, 45, "DOLLAR_AMOUNT"), # "$150,000"
            (62, 72, "DATE"),          # "01/06/2024"
        ]}
    ),
    (
        "This contract shall expire on December 31, 2026 unless renewed in writing.",
        {"entities": [
            (29, 47, "DATE"),          # "December 31, 2026"
        ]}
    ),
    (
        "GlobalTech Solutions Inc. may terminate for cause immediately upon written notice.",
        {"entities": [
            (0, 25, "PARTY"),                # "GlobalTech Solutions Inc."
            (30, 80, "TERMINATION_CLAUSE"),  # termination clause
        ]}
    ),
    (
        "The penalty for late payment is $5,000 per month starting from March 2024.",
        {"entities": [
            (31, 37, "DOLLAR_AMOUNT"),  # "$5,000"
            (63, 73, "DATE"),           # "March 2024"
        ]}
    ),
    (
        "Sunrise Ventures LLC and Pacific Bridge Corp entered this deal on 15th July 2023.",
        {"entities": [
            (0, 20, "PARTY"),   # "Sunrise Ventures LLC"
            (25, 43, "PARTY"),  # "Pacific Bridge Corp"
            (65, 80, "DATE"),   # "15th July 2023"
        ]}
    ),
]

# ─────────────────────────────────────────────
# LOAD DATA FROM DOCCANO JSON EXPORT
# ─────────────────────────────────────────────
def load_doccano_export(json_path: str) -> list:
    """
    Convert Doccano's exported JSONL format into spaCy training format.

    Doccano exports each line as:
        {"text": "...", "label": [[start, end, "LABEL"], ...]}

    We convert to:
        (text, {"entities": [(start, end, "LABEL"), ...]})
    """
    training_data = []

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            text = item["text"]
            entities = []
            for label in item.get("label", []):
                start, end, entity_label = label
                entities.append((start, end, entity_label))
            training_data.append((text, {"entities": entities}))

    logger.info(f"Loaded {len(training_data)} examples from Doccano export.")
    return training_data


# ─────────────────────────────────────────────
# CONVERT TRAINING DATA → spaCy DocBin format
# ─────────────────────────────────────────────
def convert_to_docbin(training_data: list, nlp) -> DocBin:
    """
    Convert raw training data tuples into spaCy's efficient DocBin format.
    DocBin is spaCy's binary storage — faster than raw Python lists.
    """
    db = DocBin()
    skipped = 0

    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                # char_span returns None when indices don't align with token boundaries
                logger.warning(f"  Skipping misaligned span [{start}:{end}] in: '{text[:60]}...'")
                skipped += 1
                continue
            ents.append(span)
        doc.ents = ents
        db.add(doc)

    logger.info(f"DocBin built: {len(training_data) - skipped} valid, {skipped} skipped.")
    return db


# ─────────────────────────────────────────────
# TRAIN THE NER MODEL
# ─────────────────────────────────────────────
def train_ner_model(
    training_data: list,
    model_output_dir: str,
    n_iter: int = 30,
    dropout: float = 0.35
) -> None:
    """
    Full training loop for the custom legal NER model.

    Args:
        training_data    : list of (text, {"entities": [...]}) tuples
        model_output_dir : folder to save the trained model
        n_iter           : number of training epochs (30 is a good start)
        dropout          : regularisation — prevents overfitting (0.2–0.5)
    """
    # ── 1. Create a blank English spaCy model ──
    # We start fresh — no pre-trained weights (BERT fine-tuning is in Step 3)
    nlp = spacy.blank("en")
    logger.info("Created blank English spaCy model.")

    # ── 2. Add NER pipe to the pipeline ──
    ner = nlp.add_pipe("ner", last=True)

    # ── 3. Register all custom entity labels ──
    for _, annotations in training_data:
        for _, _, label in annotations["entities"]:
            ner.add_label(label)

    logger.info(f"Registered entity labels: {ner.labels}")

    # ── 4. Split into train / validation (80/20) ──
    random.shuffle(training_data)
    split = int(len(training_data) * 0.8)
    train_data = training_data[:split]
    val_data   = training_data[split:]
    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # ── 5. Begin training ──
    # disable other pipes (none here, but good practice)
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()

        for iteration in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            examples = []

            for text, annotations in train_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)

            # Mini-batch update
            # compounding batch size: starts small, grows — good for NER
            batches = spacy.util.minibatch(examples, size=spacy.util.compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=dropout, losses=losses)

            # Print loss every 5 iterations
            if (iteration + 1) % 5 == 0:
                logger.info(f"  Iteration {iteration + 1}/{n_iter} — NER Loss: {losses.get('ner', 0):.4f}")

    # ── 6. Evaluate on validation set ──
    logger.info("\nEvaluating on validation set...")
    evaluate_model(nlp, val_data)

    # ── 7. Save the trained model ──
    os.makedirs(model_output_dir, exist_ok=True)
    nlp.to_disk(model_output_dir)
    logger.info(f"\n✓ Model saved to: {model_output_dir}")


# ─────────────────────────────────────────────
# EVALUATE: Precision, Recall, F1
# ─────────────────────────────────────────────
def evaluate_model(nlp, val_data: list) -> dict:
    """
    Compute Precision, Recall, F1 per entity label.
    These are the key metrics for NER quality.

    F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
    Target: F1 > 0.90 per label for production use.
    """
    # Count true positives, false positives, false negatives per label
    scores = {}

    for text, annotations in val_data:
        doc = nlp(text)

        # Ground truth entities
        true_ents = {(start, end, label) for start, end, label in annotations["entities"]}

        # Predicted entities
        pred_ents = {(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents}

        for label in nlp.get_pipe("ner").labels:
            if label not in scores:
                scores[label] = {"tp": 0, "fp": 0, "fn": 0}

            true_label = {e for e in true_ents if e[2] == label}
            pred_label = {e for e in pred_ents if e[2] == label}

            scores[label]["tp"] += len(true_label & pred_label)  # correctly predicted
            scores[label]["fp"] += len(pred_label - true_label)  # predicted but wrong
            scores[label]["fn"] += len(true_label - pred_label)  # missed

    # Calculate and print metrics
    print("\n" + "=" * 55)
    print(f"{'ENTITY':<25} {'PREC':>7} {'REC':>7} {'F1':>7}")
    print("=" * 55)

    overall_tp = overall_fp = overall_fn = 0
    for label, counts in scores.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"{label:<25} {precision:>7.3f} {recall:>7.3f} {f1:>7.3f}")
        overall_tp += tp; overall_fp += fp; overall_fn += fn

    # Overall micro-average
    p = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    r = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print("=" * 55)
    print(f"{'OVERALL (micro)':<25} {p:>7.3f} {r:>7.3f} {f:>7.3f}")
    print("=" * 55)
    return scores


# ─────────────────────────────────────────────
# TEST: Run prediction on new text
# ─────────────────────────────────────────────
def predict(model_path: str, text: str) -> list:
    """
    Load a saved model and run prediction on new contract text.
    Returns a list of dicts: {text, label, start, end}

    Usage:
        entities = predict("models/ner_model", "Agreement dated Jan 1 2024 between Alpha Corp...")
    """
    nlp = spacy.load(model_path)
    doc = nlp(text)

    results = []
    for ent in doc.ents:
        results.append({
            "text":  ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end":   ent.end_char
        })
    return results


# ─────────────────────────────────────────────
# MAIN — run this file to train
# ─────────────────────────────────────────────
if __name__ == "__main__":
    MODEL_DIR = "models/ner_spacy_model"

    # Option A: Use sample training data (for quick testing)
    data = SAMPLE_TRAINING_DATA

    # Option B: Load from Doccano export (for real training)
    # data = load_doccano_export("data/annotated/doccano_export.jsonl")

    # Train
    train_ner_model(data, model_output_dir=MODEL_DIR, n_iter=30, dropout=0.35)

    # Test on a new sentence
    test_text = "On January 1, 2025, Apex Financial Group and Summit Partners LLC agreed to a fee of $750,000."
    print(f"\nTest prediction on:\n'{test_text}'\n")
    entities = predict(MODEL_DIR, test_text)
    for e in entities:
        print(f"  [{e['label']}]  '{e['text']}'  (chars {e['start']}–{e['end']})")
