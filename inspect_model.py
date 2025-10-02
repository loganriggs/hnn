from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-410m')
print('\nModel architecture:')
print(model)
print('\n\nLayer names:')
for name, _ in model.named_modules():
    if name:
        print(name)
