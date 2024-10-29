## Customer Sentiment Sensitive Chatbot

A customer service chatbot that analyzes customer emotions in real time, adjusting its support style or response accordingly. If the customer appears upset, the chatbot sends calming messages, and if the customer shows a positive reaction, it automatically responds to reinforce and maintain that positivity.

---

### Environment

```python
conda create -n [env_name] python>=3.11
conda activate [env_name]
# torch version depends on your hardware.
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

---

### Traing Dataset

Amazon Review Polarity Dataset contains 34,686,770 Amazon reviews from 6,643,669 users on 2,441,053 products, from the Stanford Network Analysis Project (SNAP). This subset contains 1,800,000 training samples and 200,000 testing samples in each polarity sentiment. See more details on https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews. To install,

```bash
cd data
./source.sh
unzip archive.zip
```

Run `preprocess.ipynb` to clean data.

---

### REFERENCE 

- https://github.com/dataprofessor/llama2?tab=readme-ov-file
- https://github.com/a16z-infra/llama2-chatbot
- https://www.kaggle.com/code/micaeld/amazon-reviews-with-tensorflow
- https://github.com/fshnkarimi/Fine-tuning-an-LLM-using-LoRA