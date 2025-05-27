import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
import warnings
import psutil

warnings.filterwarnings('ignore')

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def prepare_data(tokenizer, texts, labels=None, max_len=128):
    encoded_batch = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoded_batch['input_ids']
    attention_masks = encoded_batch['attention_mask']
    labels = torch.tensor(labels) if labels is not None else None

    return TensorDataset(input_ids, attention_masks, labels)

class PROMPTEmbedding(nn.Module):
    def __init__(self,
                wte: nn.Embedding,
                n_tokens: int = 10,
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        super(PROMPTEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = \
        nn.parameter.Parameter(self.initialize_embedding(wte, n_tokens,
                                                         random_range,
                                                         initialize_from_vocab))
        def initialize_embedding(self,  wte: nn.Embedding,
                                 n_tokens: int = 10,
                                 random_range: float = 0.5,
                                 initialize_from_vocab: bool = True):
            if initialize_from_vocab:
                return self.wte.weight[:n_tokens].clone().detach()
            return torch.FloatTensor(wte.weight.size(1), n_tokens).uniform_(-random_range, random_range)

    def forward(self, tokens):
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_data = pd.read_csv('IMDB_Dataset.csv')
    #train_data = train_data.sample(25000, random_state=42)

    # Map labels to integers
    label_map = {'positive': 1, 'negative': 0}
    imdb_reviews = train_data["review"].tolist()
    sentiments = [label_map[label] for label in train_data["sentiment"].tolist()]

    # Initialize tokenizer
    model_name = "arampacha/roberta-tiny"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # Prepare dataset
    train_dataset = prepare_data(tokenizer, imdb_reviews, sentiments)

    # Split dataset
    train_size = int(0.9 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=32)
    valid_dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=32)

    # Load RoBERTa tiny model for sequence classification
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    prompt_emb = PROMPTEmbedding(model.get_input_embeddings(),
                                 n_tokens=20, initialize_from_vocab=True)
    model.set_input_embeddings(prompt_emb)
    model.to(device)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # Set up scheduler
    epochs = 10
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    for epoch_i in range(0, epochs):
        print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        t0 = time.time()
        total_train_loss = 0
        model.train()

        # Track memory usage
        mem_before_epoch = psutil.virtual_memory().used / (1024 ** 2)  # Convert to MB for more precision

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        mem_after_epoch = psutil.virtual_memory().used / (1024 ** 2)  # Convert to MB for more precision
        print(f"Epoch {epoch_i + 1} Memory used: {mem_after_epoch - mem_before_epoch:.2f} MB")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

        training_time = format_time(time.time() - t0)
        print("  Training epoch took: {:}".format(training_time))

        # Validation step
        print("Running Validation...")
        t0 = time.time()
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in valid_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs.loss
            logits = outputs.logits
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            nb_eval_steps += 1

        avg_valid_loss = total_eval_loss / len(valid_dataloader)
        avg_val_accuracy = total_eval_accuracy / nb_eval_steps
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_valid_loss))
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        print("  Validation took: {:}".format(validation_time))

    print("\nTraining complete!")

if __name__ == '__main__':
    main()
