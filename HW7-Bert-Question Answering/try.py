# Task description
"""
- Chinese Extractive Question Answering
  - Input: Paragraph + Question
  - Output: Answer

- Objective: Learn how to fine tune a pretrained model on downstream task using transformers

- Todo
    - Fine tune a pretrained chinese BERT model
    - Change hyper parameters (e.g. doc_stride)
    - Apply linear learning rate decay
    - Try other pretrained models
    - Improve preprocessing
    - Improve postprocessing
- Training tips
    - Automatic mixed precision
    - Gradient accumulation
    - Ensemble

- Estimated training time (tesla t4 with automatic mixed precision enabled)
    - Simple: 8mins
    - Medium: 8mins
    - Strong: 25mins
    - Boss: 2.5hrs
"""

# Import Package
import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from transformers import (
    BertTokenizerFast,
    AutoModel,
   AutoModelForMaskedLM,
   AutoModelForCausalLM,
   AutoModelForTokenClassification,
AdamW, BertForQuestionAnswering, BertTokenizerFast
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# Fix random seed for reproducibility
def same_seeds(seed):
    # Sets the seed for generating random numbers. Returns a torch.Generator object.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # Sets the seed for generating random numbers on all GPUs.
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(0)

# Change "fp16_training" to True to support automatic mixed precision training (fp16)
fp16_training = True

if fp16_training:
    from accelerate import Accelerator

    '''
    To quickly adapt your script to work on any kind of setup with 🤗 Accelerate just:
    1.Initialize an Accelerator object (that we will call accelerator throughout this page) 
      as early as possible in your script.
    2.Pass your dataloader(s), model(s), optimizer(s), and scheduler(s) to the prepare() method.
    3.Remove all the .cuda() or .to(device) from your code and let the accelerator handle the device placement for you.
    Step three is optional, but considered a best practice.
    '''
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device

# Documentation for the toolkit:  https://huggingface.co/docs/accelerate/


# Load Model and Tokenizer
# model_name = "bert-base-chinese"
# model_name = "luhua/chinese_pretrain_mrc_roberta_wwm_ext_large"
model_name = "hfl/chinese-bert-wwm-ext"
# model_name = "luhua/chinese_pretrain_mrc_macbert_large"
# model_name = "ckiplab/bert-base-chinese-ner"
# model_name = "ckiplab/bert-base-chinese"
# model = AutoModelForMaskedLM.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name).to(device)
# model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-ner')

'''
Transformers Tokenizer

Tokenizer 分词器，在NLP任务中起到很重要的任务，其主要的任务是将文本输入转化为模型可以接受的输入，因为模型只能输入数字，所以 tokenizer 会将文本输入转化为数值型的输入.

tokenizer 的加载和保存和 models 的方式一致，都是使用方法：from_pretrained, save_pretrained. 
这个方法会加载和保存tokenizer使用的模型结构（例如sentence piece就有自己的模型结构），以及字典。

Ex:
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.save_pretrained("directory_on_my_computer")
'''

'''
PreTrainedTokenizerFast

The PreTrainedTokenizerFast depend on the tokenizers library.

“Fast” implementation based on the Rust library Tokenizers.
The “Fast” implementations allows:
1.a significant speed-up in particular when doing batched tokenization and
2.additional methods to map between the original string (character and words) and the token space 
  (e.g. getting the index of the token comprising a given character or the span of characters corresponding 
   to a given token).

Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

Inherits from PreTrainedTokenizerBase.

Handles all the shared methods for tokenization and special tokens, as well as methods for downloading/caching/loading 
pretrained tokenizers, as well as adding tokens to the vocabulary.

This class also contains the added tokens in a unified way on top of all tokenizers so we don’t have to handle 
the specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece…).
'''

'''
BertTokenizerFast

Construct a “fast” BERT tokenizer (backed by HuggingFace’s tokenizers library). Based on WordPiece.

This tokenizer inherits from PreTrainedTokenizerFast which contains most of the main methods. 
Users should refer to this superclass for more information regarding those methods.
'''

tokenizer = BertTokenizerFast.from_pretrained(model_name)

# You can safely ignore the warning message (it pops up because new prediction heads for QA are initialized randomly)


# Read Data
'''
For 2022 data
- Training set: 31690 QA pairs
- Dev set: 4131  QA pairs
- Test set: 4957  QA pairs

For 2021 data
- Training set: 26935 QA pairs
- Dev set: 3523  QA pairs
- Test set: 3492  QA pairs

- {train/dev/test}_questions:	
  - List of dicts with the following keys:
   - id (int)
   - paragraph_id (int)
   - question_text (string)
   - answer_text (string)
   - answer_start (int)
   - answer_end (int)
- {train/dev/test}_paragraphs: 
  - List of strings
  - paragraph_ids in questions correspond to indexs in paragraphs
  - A paragraph may be used by several questions 
'''


def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        dataread = json.load(reader)
    return dataread["questions"], dataread["paragraphs"]


train_questions, train_paragraphs = \
    read_data("D:\\PycharmProjects\\HW7-Bert-Question Answering\\hw7_data2022\\hw7_train.json")
dev_questions, dev_paragraphs = \
    read_data("D:\\PycharmProjects\\HW7-Bert-Question Answering\\hw7_data2022\\hw7_dev.json")
test_questions, test_paragraphs = \
    read_data("D:\\PycharmProjects\\HW7-Bert-Question Answering\\hw7_data2022\\hw7_test.json")

# Tokenize Data

# Tokenize questions and paragraphs separately
# 「add_special_tokens」 is set to False since special tokens will be added
#  when tokenized questions and paragraphs are combined in dataset __getitem__

train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions],
                                      add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions],
                                    add_special_tokens=False)
test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions],
                                     add_special_tokens=False)

train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)


# You can safely ignore the warning message as tokenized sequences will be futher processed in datset __getitem__
# before passing to model


# Dataset and Dataloader

class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 40
        self.max_paragraph_len = 150

        # TODO: Change value of doc_stride (已完成)
        # self.doc_stride = 150
        self.doc_stride = 100

        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        # TODO: Preprocessing #####
        # Hint: How to prevent model from learning something it should not learn

        if self.split == "train":
            # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            '''
            def char_to_token(self, *args, **kwargs):
                Get the token that contains the char at the given position in the input sequence.
                Args:
                     char_pos (:obj:`int`):
                               The position of a char in the input string
                     sequence_index (:obj:`int`, defaults to :obj:`0`):
                                    The index of the sequence that contains the target char
                Returns:
                        :obj:`int`: The index of the token that contains this char in the encoded sequence
            '''
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])

            '''
            A single window is obtained by slicing the portion of paragraph containing the answer
            mid = (answer_start_token + answer_end_token) // 2
            paragraph_start = max(0, min(mid - self.max_paragraph_len // 2,
                                         len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len
            '''
            '''
            preprocessing数据前处理，之前训练的时候选取paragraph的时候总是让答案在段落的中间位置，这样会导致模型学到奇怪的东西，
            比如测试的时候输出的答案都比较偏向于中心位置，这样显然是不对的，因为测试的时候我们并不知道答案会落到哪里，
            因此在选取段落时应该随机选取，从最左边到最右边随机选取开始位置。
            '''
            start_min = max(0, answer_end_token - self.max_paragraph_len + 1)
            start_max = min(answer_start_token, len(tokenized_paragraph) - self.max_paragraph_len)
            start_max = max(start_min, start_max)
            paragraph_start = random.randint(start_min, start_max + 1)
            paragraph_end = paragraph_start + self.max_paragraph_len

            # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
            # 特殊符号意义：
            # [CLS] 标志放在第一个句子的首位，经过 BERT 得到的的表征向量 C 可以用于后续的分类任务。
            # [SEP] 标志用于分开两个输入句子，例如输入句子 A 和 B，要在句子 A，B 后面增加 [SEP] 标志。
            # [UNK]标志指的是未知字符
            # [MASK] 标志用于遮盖句子中的一些单词，将单词用 [MASK] 遮盖之后，再利用 BERT 输出的 [MASK] 向量预测单词是什么。
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start: paragraph_end] + [102]

            # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start

            # Pad sequence and obtain inputs to model
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
                attention_mask), answer_start_token, answer_end_token

        # Validation/Testing
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []

            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i: i + self.max_paragraph_len] + [102]

                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)

                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)

            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len

        return input_ids, token_type_ids, attention_mask


train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

train_batch_size = 16
doc_stride = 100

# Note: Do NOT change batch size of dev_loader / test_loader !
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)


# Function for Evaluation

def evaluate(data, output, doc_stride=doc_stride, token_type_ids=None, paragraph=None, paragraph_tokenized=None):
    # TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing
    # Hint: Open your prediction file to see what is wrong

    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]

    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)

        token_type_id = data[1][0][k].detach().cpu().numpy()
        # [CLS] + [question] + [SEP] + [paragraph] + [SEP]
        paragraph_start = token_type_id.argmax()
        paragraph_end = len(token_type_id) - 1 - token_type_id[::-1].argmax() - 1

        if start_index > end_index:
            continue

        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob

        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob:
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(data[0][0][k][start_index: end_index + 1])
            # 找到tokenized paragraph中对应的位置
            origin_start = start_index + k * doc_stride - paragraph_start
            origin_end = end_index + k * doc_stride - paragraph_start

    # Remove spaces in answer (e.g. "大 金" --> "大金")
    answer = answer.replace(' ', '')
    if '[UNK]' in answer:
        print('发现 [UNK]，这表明有文字无法编码, 使用原始文本')
        # print("Paragraph:", paragraph)
        # print("Paragraph:", paragraph_tokenized.tokens)
        print('--直接解码预测:', answer)
        # 找到原始文本中对应的位置
        if paragraph_tokenized:
            raw_start = paragraph_tokenized.token_to_chars(origin_start)[0]
            raw_end = paragraph_tokenized.token_to_chars(origin_end)[1]
            answer = paragraph[raw_start:raw_end]
            print('--原始文本预测:', answer)

    return answer


# Training

num_epoch = 1
validation = True
logging_step = 100
total_steps = num_epoch * len(train_loader)
acc_steps = 4
learning_rate = 1.5e-4 / acc_steps
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# scheduler = ExponentialLR(optimizer, gamma=0.9)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=100,
                                            num_training_steps=total_steps // acc_steps)

if fp16_training:
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

model.train()

print("Start Training ...")

for epoch in range(num_epoch):
    step = 1
    train_loss = train_acc = 0

    for batch_idx, data in enumerate(tqdm(train_loader)):
        # Load all data into GPU
        data = [i.to(device) for i in data]

        # Model inputs: in
        # put_ids, token_type_ids, attention_mask, start_positions, end_positions
        # (Note: only "input_ids" is mandatory(强制的、法定的))
        # Model outputs: start_logits, end_logits, loss
        # (return when start_positions/end_positions are provided)
        output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3],
                       end_positions=data[4])

        # Choose the most probable start position / end position
        # torch.argmax(input, dim, keepdim=False) → LongTensor
        # Returns the indices of the maximum values of a tensor across a dimension.
        start_index = torch.argmax(output.start_logits, dim=1)
        end_index = torch.argmax(output.end_logits, dim=1)

        # Prediction is correct only if both start_index and end_index are correct
        train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
        train_loss += output.loss

        if fp16_training:
            accelerator.backward(output.loss)
        else:
            output.loss.backward()

        step += 1
        if ((batch_idx + 1) % acc_steps == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # TODO: Apply linear learning rate decay #####(已完成)

        # Print training loss and accuracy over past logging step
        if step % logging_step == 0:
            print(
                f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, "
                f"acc = {train_acc / logging_step:.3f}")
            train_loss = train_acc = 0

    if validation:
        print("Evaluating Dev Set ...")
        model.eval()
        with torch.no_grad():
            dev_acc = 0
            for i, data in enumerate(tqdm(dev_loader)):
                output = model(input_ids=data[0].squeeze(dim=0).to(device),
                               token_type_ids=data[1].squeeze(dim=0).to(device),
                               attention_mask=data[2].squeeze(dim=0).to(device))
                # prediction is correct only if answer text exactly matches
                answer = evaluate(data, output, doc_stride=doc_stride,
                                  paragraph=dev_paragraphs[dev_questions[i]["paragraph_id"]],
                                  paragraph_tokenized=dev_paragraphs_tokenized[dev_questions[i]["paragraph_id"]])

                dev_acc += answer == dev_questions[i]["answer_text"]
            print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
        model.train()

# Save a model and its configuration file to the directory 「saved_model」
# i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
# Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
print("Saving Model ...")
model_save_dir = "saved_model"
model.save_pretrained(model_save_dir)

# Testing

print("Evaluating Test Set ...")

result = []

model.eval()
with torch.no_grad():
    for data in tqdm(test_loader):
        output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))

        result.append(evaluate(data, output))

result_file = "result.csv"
with open(result_file, 'w', encoding="utf-8") as f:
    f.write("ID,Answer\n")
    for i, test_question in enumerate(test_questions):
        # Replace commas in answers with empty strings (since csv is separated by comma)
        # Answers in kaggle are processed in the same way
        f.write(f"{test_question['id']},{result[i].replace(',', '')}\n")

print(f"Completed! Result is in {result_file}")
