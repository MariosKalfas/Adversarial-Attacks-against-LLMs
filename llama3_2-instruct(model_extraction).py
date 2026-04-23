import torch
from transformers import pipeline
import json
import os
from datetime import datetime


HF_TOKEN = "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" 
model_id = "meta-llama/Llama-3.2-3B-Instruct"


queries = [
    "Explain the concept of Quantum Entanglement in simple terms.",
    "Write a Python function to calculate the Fibonacci sequence.",
    "What are the main causes of the French Revolution?",
    "Translate 'Hello world' to Spanish, French, and German.",
    "List 5 healthy breakfast ideas with eggs.",
    "How does a neural network learn via backpropagation?",
    "Write a short poem about a lonely robot on Mars.",
    "What is the difference between TCP and UDP protocols?",
    "Explain how photosynthesis works in plants.",
    "Create a JSON schema for a user profile."
]

# --- ΚΥΡΙΩΣ ΠΡΟΓΡΑΜΜΑ ---
print(f"[{datetime.now().strftime('%H:%M:%S')}] Εκκίνηση διαδικασίας Model Extraction...")

try:
    # 1. Φόρτωση του Μοντέλου-Στόχου (The Victim)
    print("Φόρτωση του μοντέλου-στόχου (Victim Model)...")
    pipe = pipeline(
        "text-generation",
        model=model_id,
        token=HF_TOKEN, 
        torch_dtype=torch.float32, # CPU Mode
        device_map="auto",
        pad_token_id=50256
    )

    extracted_data = []

    print(f"\nΞεκινάει η υποκλοπή γνώσης ({len(queries)} ερωτήματα)...\n")

    # 2. Διαδικασία Εξόρυξης (Querying)
    for i, query in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] Querying: {query[:40]}...")
        
        # Κατασκευή μηνύματος
        messages = [{"role": "user", "content": query}]
        
        # Λήψη απάντησης από το θύμα
        outputs = pipe(messages, max_new_tokens=400)
        response = outputs[0]["generated_text"][-1]["content"]
        
        # 3. Αποθήκευση στο Dataset
        entry = {
            "id": i,
            "instruction": query,
            "response": response, # Η "κλεμμένη" γνώση
            "source_model": model_id
        }
        extracted_data.append(entry)

    # 4. Εγγραφή στο αρχείο (The Stolen Dataset)
    output_file = "stolen_dataset.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)

    print("\n" + "="*50)
    print("ΕΠΙΤΥΧΙΑ MODEL EXTRACTION")
    print("="*50)
    print(f"Δημιουργήθηκε το αρχείο: {output_file}")
    print(f"Συνολικά ζεύγη δεδομένων: {len(extracted_data)}")
    print("Αυτό το αρχείο μπορεί τώρα να χρησιμοποιηθεί για Fine-Tuning ενός άλλου μοντέλου.")

except Exception as e:
    print(f"\nΣφάλμα: {e}")