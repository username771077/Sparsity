
import logging
from tqdm import tqdm
import numpy as np
import torch
import random
import time
import matplotlib.pyplot as plt
from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger("transformer.log")
logging.basicConfig(level=logging.INFO)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def capture_gradients(module, grad_input, grad_output):
    if grad_output[0] is not None:
        module.grads.append(grad_output[0].detach().cpu().numpy())

def register_hooks(model):
    for layer in model.bert.encoder.layer:
        layer.intermediate.dense.grads = []
        layer.intermediate.dense.register_backward_hook(capture_gradients)
        layer.output.dense.grads = []
        layer.output.dense.register_backward_hook(capture_gradients)

def plot_gradients(grads, title):
    for i, grad in enumerate(grads):
        if grad:            
            grad_first_element = grad[0]            
            grad_mean = np.mean(grad_first_element, axis=0)
            plt.imshow(grad_mean.reshape(-1, 1), cmap='gray', aspect='auto')
            plt.title(f'{title} Layer {i}')
            plt.colorbar()
            plt.show()

def calculate_sparsity(grads):
    sparsity = []
    for grad in grads:
        if grad:
            sparsity.append(np.mean(grad[0] == 0))
    return sparsity

def load_and_cache_examples(tokenizer, dataset_name, config_name, split, block_size=512):
    dataset = load_dataset(dataset_name, config_name, split=split)
    dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=block_size), batched=True)
    dataset.set_format(type='torch', columns=['input_ids'])
    return dataset

def compute_average_length(dataset):
    lengths = [len(sample['input_ids']) for sample in dataset]
    return np.mean(lengths)

class Args:
    def __init__(self):
        self.output_dir = "./results"
        self.dataset_name = "wikitext"
        self.config_name = "wikitext-103-v1"
        self.train_split = "train"
        self.eval_split = "validation"
        self.per_gpu_train_batch_size = 2
        self.per_gpu_eval_batch_size = 2
        self.n_gpu = 1
        self.gradient_accumulation_steps = 1
        self.num_train_epochs = 3  
        self.learning_rate = 1e-4 
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.warmup_steps = 0
        self.max_grad_norm = 1.0
        self.logging_steps = 1
        self.save_steps = 1
        self.max_steps = 10  
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = False 

def train_model(args, model, tokenizer, train_dataset, valid_dataset):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.per_gpu_eval_batch_size)

    t_total = min(len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs, args.max_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_total)

    set_seed(args.seed)
    model.zero_grad()
    model.train()
    register_hooks(model)

    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0

    start_time = time.time()
    forward_times = []
    backward_times = []

    logger.info(f"Training batch size: {args.train_batch_size}")
    logger.info(f"Total optimization steps: {t_total}")

    for epoch in range(int(args.num_train_epochs)):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", total=args.max_steps)
        for step, batch in enumerate(epoch_iterator):
            inputs = batch['input_ids'].to(args.device)
            labels = batch['input_ids'].to(args.device)

            forward_start_time = time.time()
            outputs = model(inputs, labels=labels)
            forward_time = time.time() - forward_start_time
            forward_times.append(forward_time)

            loss = outputs.loss

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            backward_start_time = time.time()
            loss.backward()
            backward_time = time.time() - backward_start_time
            backward_times.append(backward_time)

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info(f'Global step: {global_step}, Loss: {(tr_loss - logging_loss)/args.logging_steps}')
                    logging_loss = tr_loss

                if global_step >= args.max_steps:
                    break

        if global_step >= args.max_steps:
            break

    end_time = time.time()
    total_training_time = end_time - start_time

    avg_forward_time = np.mean(forward_times)
    avg_backward_time = np.mean(backward_times)

    return global_step, tr_loss / global_step, total_training_time, avg_forward_time, avg_backward_time

def main():
    args = Args()

    model_name = 'bert-base-uncased'
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.to(args.device)

    train_dataset = load_and_cache_examples(tokenizer, args.dataset_name, args.config_name, args.train_split)
    valid_dataset = load_and_cache_examples(tokenizer, args.dataset_name, args.config_name, args.eval_split)

    avg_train_len = compute_average_length(train_dataset)
    avg_valid_len = compute_average_length(valid_dataset)

    logger.info(f"Average training sample length: {avg_train_len}")
    logger.info(f"Average validation sample length: {avg_valid_len}")

    global_step, tr_loss, total_training_time, avg_forward_time, avg_backward_time = train_model(args, model, tokenizer, train_dataset, valid_dataset)

    logger.info(f"Training completed. Global step: {global_step}, Avg loss: {tr_loss}, Total training time: {total_training_time:.2f} seconds")
    logger.info(f"Average forward pass time: {avg_forward_time:.6f} seconds")
    logger.info(f"Average backward pass time: {avg_backward_time:.6f} seconds")

    intermediate_grads = [layer.intermediate.dense.grads for layer in model.bert.encoder.layer]
    output_grads = [layer.output.dense.grads for layer in model.bert.encoder.layer]

    intermediate_sparsity = calculate_sparsity(intermediate_grads)
    output_sparsity = calculate_sparsity(output_grads)

    logger.info(f"Intermediate Layer Sparsity: {intermediate_sparsity}")
    logger.info(f"Output Layer Sparsity: {output_sparsity}")

    plot_gradients(intermediate_grads, 'Intermediate Gradients')
    plot_gradients(output_grads, 'Output Gradients')

if __name__ == "__main__":
    main()
