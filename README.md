# DeepSeek-R1-Medical-COT

This project fine-tunes the [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) model using a **medical chain-of-thought (CoT)** dataset for clinical reasoning. Training is performed using the [Unsloth](https://github.com/unslothai/unsloth) framework on **Kaggle T4 GPU**, optimized with 4-bit quantization and parameter-efficient LoRA adapters.

---

## 🚀 Outcome
Successfully fine-tuned the `DeepSeek-R1-Distill-Llama-8B` model using a medical CoT dataset (FreedomIntelligence/medical-o1-reasoning-SFT) on Kaggle T4 GPU. The model was able to:
- Improve step-by-step reasoning quality in complex medical diagnosis tasks.
- Generate structured CoT outputs that aligned more closely with the desired medical format.
- Reduce verbosity and increase precision in responses post fine-tuning.

---


## 🎯 Objective

To train a reasoning model that can:
- Interpret medical questions
- Think step-by-step using `<think>...</think>` tags
- Produce logical, expert-level responses

---

## 🧠 Dataset

- **Source:** [`FreedomIntelligence/medical-o1-reasoning-SFT`](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- **Subset Used:** First 500 samples
- **Fields Used:**
  - `Question`: Medical case scenario
  - `Complex_CoT`: Step-by-step reasoning
  - `Response`: Final answer

---

## 🧪 Prompt Format

A custom prompt structure was used to guide the model into thinking step-by-step before giving an answer.

```text
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{Question}

### Response:
<think>
{Complex_CoT}
</think>
{Response}
```
## ⚙️ Training Configuration
| Parameter                     | Value | Reason                                      |
| ----------------------------- | ----- | ------------------------------------------- |
| `LoRA r`                      | 16    | Standard rank size for efficient adaptation |
| `lora_alpha`                  | 16    | Scales low-rank updates                     |
| `train_batch_size`            | 2     | Fits into Kaggle T4 memory                  |
| `gradient_accumulation_steps` | 4     | Effective batch size = 8                    |
| `learning_rate`               | 2e-4  | Optimized for PEFT and small dataset        |
| `max_steps`                   | 60    | Sufficient for 500-sample dataset           |
| `max_seq_length`              | 2048  | Long enough for question + CoT + answer     |



- Trainable Parameters: 41.94M

- Total Parameters: 8072.20M

- % Trainable: 0.52%

## 📈 Example Comparison (Before vs After Fine-Tuning)

#### ✅Question

> A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or sneezing but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings, what would cystometry most likely reveal about her residual volume and detrusor contractions?


#### ❌ Before Fine-Tuning
```text
<think>
Okay, so I need to figure out what cystometry would show for this 61-year-old woman. Let me start by breaking down the information given.

First, the patient has a history of involuntary urine loss when she coughs or sneezes. That makes me think about stress urinary incontinence. I remember that stress incontinence usually occurs when the urethral sphincter isn't closing properly during activities that increase abdominal pressure, like coughing. But she doesn't leak at night, so it's not likely to be urge incontinence, which often happens during sleep.

She undergoes a gynecological exam and a Q-tip test. I'm a bit fuzzy on what the Q-tip test entails. From what I recall, the Q-tip is a small catheter used to measure urethral pressure and resistance. The test probably involved inserting it into her urethra to check for any obstructions or to evaluate the sphincter's function.

So, during the Q-tip test, if the urethral resistance is low, that could mean the sphincter isn't working properly, which would explain the stress incontinence. If the resistance is normal, maybe the issue is elsewhere, but given her history, I'm leaning towards a sphincter problem.

Now, thinking about cystometry. I know that cystometry is a test where they fill the bladder and measure how much it holds (residual volume) and the contractions of the detrusor muscle when it's filled. 

If the Q-tip test showed low urethral resistance, that would support a diagnosis of stress incontinence. So during cystometry, the residual volume might be normal, but the detrusor contractions could be weak or ineffective. This would mean that when the bladder tries to contract, it doesn't push the urine out effectively, leading to the involuntary leakage.

Alternatively, if the Q-tip showed normal resistance but other findings, maybe the issue is with the detrusor muscle itself, like a neurogenic problem. In that case, the residual volume might be normal, but the detrusor contractions would be absent or weak, leading to retention or incontinence.

But considering the patient's history and the Q-tip results pointing towards sphincter weakness, I think the cystometry would show normal residual volume but weak detrusor contractions. This would align with stress incontinence due to sphincter insufficiency, where the bladder can hold normal amounts but can't squeeze them out effectively.

I should also consider other possibilities, like if the patient has an overactive bladder, but since she doesn't leak at night, that's less likely. The key here is that the leakage is during activities that increase abdominal pressure, which points to the sphincter issue.

So, putting it all together, the cystometry would likely show a normal residual volume but weak detrusor contractions, indicating that the main problem is with the urethral sphincter not closing properly during those activities.
</think>

Based on the patient's history and the findings from the gynecological exam and Q-tip test, cystometry is expected to reveal the following:

1. **Residual Volume**: Normal. This indicates that the bladder can hold a typical amount of urine without leakage.

2. **Detrusor Contractions**: Weak or ineffective. The detrusor muscle's ability to contract and push urine out is diminished, which aligns with a diagnosis of stress urinary incontinence due to urethral sphincter insufficiency.

This conclusion supports the presence of stress incontinence, where the sphincter fails to close properly during activities like coughing, leading to involuntary leakage, while the bladder's storage capacity remains normal.<｜end▁of▁sentence｜>
```

#### ✅ After Fine-Tuning
```text
<think>
Okay, let's see. We have a 61-year-old woman who has been dealing with this annoying urinary issue for a while. She's had involuntary urine loss whenever she coughs or sneezes, but she's never had any leakage at night. That’s interesting because it suggests there might be something wrong with her bladder mechanics during activities, but not at rest.

Now, she's had a gynecological exam and a Q-tip test. The Q-tip test is a standard diagnostic tool for bladder problems, especially in cases like this. It involves inserting a catheter and then letting the patient cough, and then checking the position of the Q-tip as it exits the urethra. The key here is to see if the Q-tip is pulled back, which would indicate that the bladder neck is not properly closing, leading to incontinence.

If the Q-tip test shows the tip is pulled back, it usually means the bladder neck is not closing well. This can lead to a condition known as urethral incompetence, where the urethra doesn't fully close. This is often linked to a condition called urethral sphincter deficiency.

Now, thinking about what we might see in a cystometry. Cystometry is a test where we fill the bladder and then let the patient empty it. It's used to assess how the bladder behaves when it's full. If there's urethral sphincter deficiency, we’d expect to see some kind of leakage. Specifically, we might see that the residual volume is increased, meaning the bladder doesn't empty completely. This residual volume is usually a sign of impaired sphincter function.

Also, during the test, we’d expect to see detrusor contractions. Detrusor contractions are basically the contractions that push the urine out of the bladder. They’re usually normal in healthy bladders. However, if there's urethral sphincter deficiency, we might see some abnormal contractions, or maybe even a change in the pattern of contractions, which could indicate that the bladder is not functioning as it should.

So putting it all together, given that the Q-tip test suggests urethral sphincter deficiency, we’d expect the cystometry to show an increased residual volume and possibly some abnormal detrusor contractions. This would confirm that there's an issue with the sphincter function, contributing to the involuntary urine loss she's experiencing.
</think>
Based on the findings from the Q-tip test, which suggest urethral sphincter deficiency, the cystometry would most likely reveal an increased residual volume and possibly abnormal detrusor contractions. This indicates impaired sphincter function, which aligns with the clinical presentation of involuntary urine loss during activities like coughing or sneezing.<｜end▁of▁sentence｜>

```

## 📂 Output Files
All outputs are saved in the `DeepSeek-R1-Medical-COT/` folder:
| File                        | Description                     |
| --------------------------- | ------------------------------- |
| `adapter_model.safetensors` | Fine-tuned LoRA adapter weights |
| `adapter_config.json`       | LoRA configuration              |
| `tokenizer.json`            | Tokenizer vocabulary            |
| `tokenizer_config.json`     | Tokenizer settings              |
| `special_tokens_map.json`   | Special token mappings          |
| `chat_template.jinja`       | Prompt formatting template      |


## 📊 Training Metrics

#### Training was tracked using Weights & Biases (wandb):
> ![alt text](image-1.png)
![alt text](image-2.png)

Training completed in ~44 minutes on Kaggle’s free T4 GPU. The training loss showed consistent downward trends.

## 🧑‍💻 Author
Developed by: [Prathamesh Pawar](https://github.com/Prathameshppawar)

License: Apache-2.0

## 📝 Notes

- Model is fine-tuned and stored locally only

- No public upload to Hugging Face
