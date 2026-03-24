from transformers import pipeline

# Load pretrained text generator
generator = pipeline("text-generation", model="gpt2")

# Input
seed = input("Enter seed text: ")

# Generate text
output = generator(seed, max_length=50, num_return_sequences=1)

print("\nGenerated Text:\n")
print(output[0]['generated_text'])