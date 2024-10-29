from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import numpy as np

from src.dataset import AmazonDataset
from src.utils import set_seed

def train():
    
    set_seed(42)
    model_checkpoint = 'bert-base-cased'
    
    # hyperparameters
    lr = 1e-3
    batch_size = 128
    num_epochs = 10
    
    tokenizer =  AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
    # tokenizer.truncation_side = "left"

    dataset = AmazonDataset(tokenizer = tokenizer,
                            max_length = 128,
                            max_train_samples = 10000,
                            max_test_samples = 1000)    
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, 
        num_labels = 2,
        id2label = dataset.id2label, 
        label2id = dataset.label2id
    )
    
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    # data_collator =  default_data_collator
    
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=4,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules = ['query'])
    
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

        
    # define training arguments
    training_args = TrainingArguments(
        output_dir = "checkpoint/"+model_checkpoint + "-lora-text-classification",
        learning_rate=lr,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # creater trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset = dataset.tokenized_dataset['train'],
        eval_dataset = dataset.tokenized_dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        )

    # train model
    trainer.train()
    trainer.save_model()
    
    # eval
    trainer.evaluate()
    
if __name__ == '__main__':
    train()