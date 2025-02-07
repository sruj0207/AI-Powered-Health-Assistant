import sacremoses
import streamlit as st
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,Trainer,TrainingArguments

# Reduces memory usage

# Load the PubMedQA dataset
dataset = load_dataset("pubmed_qa", "pqa_labeled")

# Print to check dataset structure
print(dataset)

model_name = "microsoft/BioGPT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# Convert to bfloat16 for reduced memory usage
model.to(torch.bfloat16)  



device = torch.device("cpu")  # Force CPU usage
model.to(device)
def preprocess_function(examples):
    # Extract text data correctly from dictionary
    questions = examples["question"]  # Ensure this is a list of strings
    contexts = examples["context"]  # Ensure this is a list of strings
    long_answers = examples["long_answer"]  # Labels (target text)

    # Convert dictionary fields to strings if needed
    questions = [q if isinstance(q, str) else "" for q in questions]
    contexts = [c if isinstance(c, str) else "" for c in contexts]
    long_answers = [a if isinstance(a, str) else "" for a in long_answers]

    # Combine question + context as input
    inputs = [q + " " + c for q, c in zip(questions, contexts)]
    
    # Tokenize input and labels
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=256)
    labels = tokenizer(long_answers, truncation=True, padding="max_length", max_length=256)

    model_inputs["labels"] = labels["input_ids"]  # Assign tokenized labels
    return model_inputs
# Check available dataset splits
print(dataset)

# ✅ Fix: Create a validation split manually if missing
from datasets import DatasetDict

# ✅ Fix: Convert dataset back to DatasetDict
if "validation" not in dataset:
    dataset_split = dataset["train"].train_test_split(test_size=0.1)  # 90% train, 10% validation
    dataset = DatasetDict({
        "train": dataset_split["train"],
        "validation": dataset_split["test"]  # Rename test as validation
    })


# ✅ Now you can access dataset["train"] and dataset["validation"]


# Apply tokenization

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["pubid", "question", "context", "long_answer", "final_decision"])



tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["question", "final_decision"])


training_args = TrainingArguments(
    output_dir="./biogpt_finetuned",
    eval_strategy="epoch",  # ✅ Use 'eval_strategy' instead of 'evaluation_strategy'
    logging_strategy="steps",
    logging_steps=500,
    save_total_limit=1,  
    per_device_train_batch_size=1,  
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    gradient_accumulation_steps=16,  
    push_to_hub=False,
    report_to="none"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()

# Save the model and tokenizer
model.save_pretrained("./biogpt_finetuned")
tokenizer.save_pretrained("./biogpt_finetuned")
print("Fine-tuning complete! Model saved.")

@st.cache_resource
def load_finetuned_model():
    model = AutoModelForCausalLM.from_pretrained("./biogpt_finetuned")
    tokenizer = AutoTokenizer.from_pretrained("./biogpt_finetuned")
    model.to(torch.bfloat16)  # Convert to bfloat16 for efficiency
    model.to(device)  # Move model to CPU
    return model, tokenizer

# Load model once
model, tokenizer = load_finetuned_model()

# Streamlit UI
st.title("🩺 BioGPT Health Assistant Chatbot")
st.markdown("Ask any medical question, and BioGPT will generate a response.")

# User input
user_input = st.text_area("Enter your health-related question:", "")

if st.button("Generate Response"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating response..."):
            # Tokenize input
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to CPU

            # Generate response
            output = model.generate(**inputs, max_length=100)
            response = tokenizer.decode(output[0], skip_special_tokens=True)

            # Display the response
            st.success("### 💡 BioGPT's Response:")
            st.write(response)