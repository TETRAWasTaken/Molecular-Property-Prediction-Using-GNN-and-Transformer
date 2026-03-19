from transformers import AutoTokenizer

def test_tokenizer():
    # 1. Load the pre-trained ChemBERTa tokenizer
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Define a test SMILES string (e.g., Aspirin)
    sample_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    print(f"\nOriginal SMILES: {sample_smiles}")

    # 3. See how it splits the string into sub-word tokens
    # This is what the model actually "reads"
    tokens = tokenizer.tokenize(sample_smiles)
    print(f"\nTokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")

    # 4. Generate the exact dictionary needed for the model
    # We use a short max_length for this test so the output isn't massive
    encoded = tokenizer(
        sample_smiles,
        padding='max_length',
        truncation=True,
        max_length=20,
        return_tensors='pt'  # Return PyTorch tensors
    )

    print("\n--- Tensor Outputs ---")
    print(f"Input IDs shape: {encoded['input_ids'].shape}")
    print(f"Input IDs:\n{encoded['input_ids']}")

    print(f"\nAttention Mask shape: {encoded['attention_mask'].shape}")
    print(f"Attention Mask:\n{encoded['attention_mask']}")

    # 5. Reverse the process (decode the IDs back to text)
    # Notice the special tokens <s> (start) and </s> (end) and <pad>
    decoded_text = tokenizer.decode(encoded['input_ids'][0])
    print(f"\nDecoded Text (Notice the special tokens):\n{decoded_text}")


if __name__ == "__main__":
    test_tokenizer()