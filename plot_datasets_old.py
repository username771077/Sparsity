import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

def make_dataset(model_path_name, dataset_path, dataset_name, max_length):
    dataset = load_dataset(dataset_path, dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path_name)
    
    def tokenize_function(examples):
        if dataset_name == 'cola':
            return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=max_length)
        else:
            return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length', max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return tokenized_dataset

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=1)
    return {"accuracy": (predictions == labels).astype(float).mean().item()}

def train_and_evaluate(model_name, dataset_name, output_dir):
    dataset = make_dataset(model_name, 'glue', dataset_name, max_length=128)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        no_cuda=False if torch.cuda.is_available() else True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    return trainer

def print_histogram(trainer, dataset_name):
    model = trainer.model
    grad_sparsity_intermediate = [[] for _ in range(12)]
    grad_sparsity_output = [[] for _ in range(12)]
    
    # Register hooks to capture gradients
    def capture_intermediate_gradients(module, grad_input, grad_output):
        if grad_output[0] is not None:
            grad_sparsity_intermediate[module.layer_idx].append(grad_output[0].detach().cpu().numpy())
        print(f"Captured intermediate gradients for layer {module.layer_idx}")

    def capture_output_gradients(module, grad_input, grad_output):
        if grad_output[0] is not None:
            grad_sparsity_output[module.layer_idx].append(grad_output[0].detach().cpu().numpy())
        print(f"Captured output gradients for layer {module.layer_idx}")
    
    for i in range(12):
        model.bert.encoder.layer[i].intermediate.dense.layer_idx = i
        model.bert.encoder.layer[i].output.dense.layer_idx = i
        
        model.bert.encoder.layer[i].intermediate.dense.register_full_backward_hook(capture_intermediate_gradients)
        model.bert.encoder.layer[i].output.dense.register_full_backward_hook(capture_output_gradients)
    model.train()
    for batch in trainer.get_train_dataloader():
        inputs = {k: v.to(trainer.args.device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        break 
    
    avg_sparsity_intermediate = [np.mean([np.mean(g == 0) for g in layer_grads]) for layer_grads in grad_sparsity_intermediate if layer_grads]
    avg_sparsity_output = [np.mean([np.mean(g == 0) for g in layer_grads]) for layer_grads in grad_sparsity_output if layer_grads]
    layers = min(len(avg_sparsity_intermediate), len(avg_sparsity_output))
    
    print(f"Average sparsity intermediate: {avg_sparsity_intermediate}")
    print(f"Average sparsity output: {avg_sparsity_output}")
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))r
    ax[0].bar(range(layers), avg_sparsity_intermediate[:layers], color='b', alpha=0.6, label='Intermediate Layers')
    ax[1].bar(range(layers), avg_sparsity_output[:layers], color='r', alpha=0.6, label='Output Layers')

    ax[0].set_title(f'Gradient Sparsity for Intermediate Layers ({dataset_name})')
    ax[0].set_xlabel('Layer')
    ax[0].set_ylabel('Sparsity')
    ax[0].legend()
    
    ax[1].set_title(f'Gradient Sparsity for Output Layers ({dataset_name})')
    ax[1].set_xlabel('Layer')
    ax[1].set_ylabel('Sparsity')
    ax[1].legend()

    plt.suptitle(f'Gradient Sparsity for {dataset_name}')
    plt.show()

trainer = train_and_evaluate('bert-base-uncased', 'cola', './results/cola')
print_histogram(trainer, 'CoLA')
