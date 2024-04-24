from collections import defaultdict
import openai
import time
import re
import json
    
def load_conll_into_sentences(in_tsv_file):

    with open(in_tsv_file, 'r', encoding='utf-8') as in_tsv_reader:
        sentences = []
        sentence = []
        labels = []
        for line in in_tsv_reader:
            line = line.rstrip()
            if line == '':
                if len(sentence) > 0:
                    sentences.append([sentence, labels])
                    sentence = []
                    labels = []
            else:
                sentence.append(line.split('\t')[0])
                labels.append(line.split('\t')[-1])
        if len(sentence) > 0:
            sentences.append([sentence, labels])
    
    return sentences

def load_relation_into_sentences(in_tsv_file):

    with open(in_tsv_file, 'r', encoding='utf-8') as in_tsv_reader:
        sentences = []

        for line in in_tsv_reader:

            tks = line.rstrip().split('\t')
            if len(tks) == 3:
                index = tks[0]
                sentence = tks[1]
                sentences.append((index, sentence))

    return sentences
    
def convert_token_label_2_sent_and_html(tokens, labels):
    
    index = 0

    sent = ''
    html = ''
    num_tokens = len(tokens)
    pre_label = 'O'

    while index < num_tokens:
        if labels[index] == 'B':
            if pre_label != 'O':
                html += ' </span> <span> ' + tokens[index]
            else:
                html += ' <span> ' + tokens[index]
            pre_label = 'B'
            sent += ' ' + tokens[index]
        elif labels[index] == 'O':
            if pre_label != 'O':
                html += ' </span> <span> ' + tokens[index]
            else:
                html += ' ' + tokens[index]
            pre_label = 'O'
            sent += ' ' + tokens[index]
        elif labels[index] == 'I':
            html += ' ' + tokens[index]
            pre_label = 'I'
            sent += ' ' + tokens[index]
        index += 1
    
    if pre_label != 'O':
        html += ' </span>'

    return sent.strip(), html.strip()
    
def load_ner_examples(in_train_json, in_similarity_json):

    with open(in_train_json, 'r', encoding='utf-8') as in_train_reader:
        train_data = json.load(in_train_reader)

    with open(in_similarity_json, 'r', encoding='utf-8') as in_similarity_reader:
        similarity_data = json.load(in_similarity_reader)

    ranked_examples = {}

    index_2_sent_and_html = {}

    for example in train_data:
        index  = str(example['index'])
        labels = example['gold']
        tokens = example['sentence']
        sent, html = convert_token_label_2_sent_and_html(tokens, labels)

        index_2_sent_and_html[index] = (sent, html)

    for obj in similarity_data:
        for index, value_list in obj.items():
            ranked_examples[index] = [index_2_sent_and_html[val['train_index']] for val in value_list]

    return ranked_examples

def load_re_examples(in_train_json, in_similarity_json):
    
    with open(in_train_json, 'r', encoding='utf-8') as in_train_reader:
        train_data = json.load(in_train_reader)

    with open(in_similarity_json, 'r', encoding='utf-8') as in_similarity_reader:
        similarity_data = json.load(in_similarity_reader)

    ranked_examples = {}

    index_2_sent_and_label = {}

    for example in train_data:
        index  = str(example['index'])
        label = example['gold']
        sent  = example['sentence']

        index_2_sent_and_label[index] = (sent, label)

    for obj in similarity_data:
        for index, value_list in obj.items():
            ranked_examples[index] = [index_2_sent_and_label[val['train_index']] for val in value_list]

    return ranked_examples
    
def load_hoc_examples(in_train_json, in_similarity_json):
    
    with open(in_train_json, 'r', encoding='utf-8') as in_train_reader:
        train_data = json.load(in_train_reader)

    with open(in_similarity_json, 'r', encoding='utf-8') as in_similarity_reader:
        similarity_data = json.load(in_similarity_reader)

    ranked_examples = {}
    
    index_2_sent_and_label = {}

    for example in train_data:
        hoc_dict = {"activating invasion and metastasis": 0,
                "sustaining proliferative signaling": 0,
                "resisting cell death": 0,
                "cellular energetics": 0,
                "genomic instability and mutation": 0,
                "evading growth suppressors": 0,
                "inducing angiogenesis": 0,
                "enabling replicative immortality": 0,
                "avoiding immune destruction": 0,
                "tumor promoting inflammation": 0}

        index  = str(example['index'])
        label = example['gold']
        sent  = example['sentence']
        
        for _label in label.split(';'):
            hoc_dict[_label.strip()] = 1

        index_2_sent_and_label[index] = (sent, str(hoc_dict))

    for obj in similarity_data:
        for index, value_list in obj.items():
            ranked_examples[index] = [index_2_sent_and_label[val['train_index']] for val in value_list]

    return ranked_examples
    
def load_litcoin_examples(in_train_json, in_similarity_json):
    
    with open(in_train_json, 'r', encoding='utf-8') as in_train_reader:
        train_data = json.load(in_train_reader)

    with open(in_similarity_json, 'r', encoding='utf-8') as in_similarity_reader:
        similarity_data = json.load(in_similarity_reader)

    ranked_examples = {}
    
    index_2_sent_and_label = {}

    for example in train_data:
        litcoin_dict = {"Treatment": 0,
                        "Diagnosis": 0,
                        "Prevention": 0,
                        "Mechanism": 0,
                        "Transmission": 0,
                        "Epidemic Forecasting": 0,
                        "Case Report": 0
                        }

        index  = str(example['index'])
        label = example['gold']
        sent  = example['sentence']
        
        for _label in label.split(';'):
            litcoin_dict[_label.strip()] = 1

        index_2_sent_and_label[index] = (sent, str(litcoin_dict))

    for obj in similarity_data:
        for index, value_list in obj.items():
            ranked_examples[index] = [index_2_sent_and_label[val['train_index']] for val in value_list]

    return ranked_examples


def load_tsv_to_token_label_list(in_tsv_file):

    all_tokens, all_labels, sentences = [], [], []
    sent_id = 0
    with open(in_tsv_file, 'r', encoding='utf8') as reader:
        for line in reader:

            line = line.rstrip()

            if line == '':
                continue
            
            tokens =[]
            labels = []
            tks = line.split('\t')
            label = 'O'
            if len(tks) > 1:
                
                sentences.append(tks[1])
                tks[1] = re.sub(r'\s*</span>\s*', ' </span> ', re.sub(r'\s*<span>\s*((disease|chemical):\s*)?', ' <span> ', tks[1], flags=re.IGNORECASE), flags=re.IGNORECASE)
                for token in tks[1].split(' '):
                    
                    token = token.strip()

                    if token == '<span>':
                        label = 'B'
                    elif token == '</span>':
                        label = 'O'
                    elif label == 'B':
                        tokens.append(token)
                        labels.append(label)
                        label = 'I'
                    elif label == 'I':
                        tokens.append(token)
                        labels.append(label)
                    elif label == 'O':
                        tokens.append(token)
                        labels.append(label)
            else:
                sentences.append(tks[0])
                for token in tks[0].split(' '):                
                    token = token.strip()
                    tokens.append(token)
                    labels.append('O')
            #for token, label in zip(tokens, labels):
            #    print(token, label)
            #print(len(all_tokens), len(all_labels), len(sentences))
            all_tokens.append(tokens)
            all_labels.append(labels)
    return all_tokens, all_labels, sentences

def convert_token_label_2_sent_ne_tuples_list(all_tokens, all_labels):
    
    sent_id_2_ne_tuples_list = []

    for i, (tokens, labels) in enumerate(zip(all_tokens, all_labels)):
        ne_list = set()

        index = 0
        start = -1
        ne_text = ''
        num_tokens = len(tokens)
        num_chars = 0
        while index < num_tokens:
            if labels[index] == 'B':
                if start != -1:
                    ne_list.add((start, num_chars, ne_text))
                    start = -1
                    ne_text = ''
                else:
                    start = num_chars
                    ne_text = tokens[index]
            elif labels[index] == 'O':
                if start != -1:
                    ne_list.add((start, num_chars, ne_text))
                    start = -1
                    ne_text = ''
            elif labels[index] == 'I':
                ne_text += ' ' + tokens[index]
                
            num_chars += len(tokens[index])
            index += 1
        if start != -1:
            ne_list.add((start, num_chars, ne_text))
        sent_id_2_ne_tuples_list.append(ne_list)

    return sent_id_2_ne_tuples_list

def convert_ne_list_2_iob2(gold_tokens, ne_lists):

    iob2_labels = []
    for index, (tokens, ne_list) in enumerate(zip(gold_tokens, ne_lists)):

        start = 0
        labels = ['O' for _ in range(len(tokens))]
        
        if len(ne_list) != 0:

            for ne in ne_list:
                start, end, ne_text = ne
                
                offset = 0
                begin_ne = False
                for i, token in enumerate(tokens):
                    if offset == start:
                        labels[i] = 'B'
                        begin_ne = True
                    elif begin_ne and start < offset and offset + len(token) <= end:
                        labels[i] = 'I'
                    else:
                        begin_ne = False
                    offset += len(token)

        iob2_labels.append(labels)
    
    return iob2_labels

def convert_2_prmopt(sentence, 
                     shot, 
                     task,
                     ranked_examples, 
                     index):

    prompt_text = ''
    if task == 'ncbi_disease':
        prompt_text = '''TASK: the task is to extract disease entities in a sentence.

INPUT: the input is a sentence.

OUTPUT: the output is an HTML that highlights all the disease entities with <span> ... </span> in the sentence.

The following example format is provided:\n\n'''
        for i in range(0, shot):
            index = str(index)
            prompt_text += 'Example-' + str(i+1) + ' Sentence: ' + ranked_examples[index][i][0] + '\n\nHTML: ' + ranked_examples[index][i][1] + '\n\n'

        prompt_text += '''Sentence: ''' + sentence + '\n\nHTML: '

    elif task == 'bc5cdr_chemical':
        prompt_text = '''TASK: the task is to extract chemical entities in a sentence.

INPUT: the input is a sentence.

OUTPUT: the output is an HTML that highlights all the chemical entities with <span> ... </span> in the sentence.

The following example format is provided:\n\n'''
        for i in range(0, shot):
            index = str(index)
            prompt_text += 'Example-' + str(i+1) + ' Sentence: ' + ranked_examples[index][i][0] + '\n\nHTML: ' + ranked_examples[index][i][1] + '\n\n'

        prompt_text += '''Sentence: ''' + sentence + '\n\nHTML: '

    elif task == 'chemprot':
        prompt_text = '''TASK: the task is to classify relations between a chemical and a gene for a sentence. 

INPUT: the input is a sentence where the chemical is labeled as @CHEMICAL$ and the gene is labeled as @GENE$ accordingly in a sentence. 

OUTPUT: your task is to select one out of the six types of relations ('CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9', and 'false') for the gene and chemical without any explanation or other characters:

CPR:3, which includes UPREGULATOR, ACTIVATOR, and INDIRECT UPREGULATOR

CPR:4, which includes DOWNREGULATOR, INHIBITOR ,and INDIRECT DOWNREGULATOR

CPR:5, which includes AGONIST, AGONIST ACTIVATOR, and AGONIST INHIBITOR

CPR:6, which includes ANTAGONIST

CPR:9, which includes SUBSTRATE, PRODUCT OF and SUBSTRATE PRODUCT OF

false, which indicates no relations

The following examples are provided:\n\n'''

        for i in range(0, shot):
            prompt_text += 'Example-' + str(i+1) + ' Q: ' + ranked_examples[index][i][0] + '\n\nA: ' + ranked_examples[index][i][1] + '\n\n'

        prompt_text += '''Q: ''' + sentence + '\n\nA: '

    elif task == 'ddi':
        prompt_text = '''TASK: the task is to classify relations between two drugs for a sentence. 

INPUT: the input is a sentence where the drugs are labeled as @DRUG$. 

OUTPUT: your task is to select one out of the five types of relations ('DDI-effect', 'DDI-mechanism', 'DDI-advise', 'DDI-false', and 'DDI-int') for the drugs without any explanation or other characters:

DDI-mechanism: This type is used to annotate DDIs that are described by their PK mechanism (e.g. Grepafloxacin may inhibit the metabolism of theobromine)

DDI-effect: This type is used to annotate DDIs describing an effect (e.g. In uninfected volunteers, 46% developed rash while receiving SUSTIVA and clarithromycin) or a PD mechanism (e.g. Chlorthalidone may potentiate the action of other antihypertensive drugs)

DDI-advise: This type is used when a recommendation or advice regarding a drug interaction is given (e.g. UROXATRAL should not be used in combination with other alpha-blockers)

DDI-int: This type is used when a DDI appears in the text without providing any additional information (e.g. The interaction of omeprazole and ketoconazole has been established)

DDI-false, This type is used when no DDI relation appears

The following examples are provided:\n\n'''

        for i in range(0, shot):
            prompt_text += 'Example-' + str(i+1) + ' Q: ' + ranked_examples[index][i][0] + '\n\nA: ' + ranked_examples[index][i][1] + '\n\n'

        prompt_text += '''Q: ''' + sentence + '\n\nA: '

    elif task == 'hoc':
    
        prompt_text = '''***TASK***

The task is to perform a semantic classification of the article according to the hallmarks of cancer based on its abstract.

***INPUT***

The input is an abstract text.

***DOCUMENTATION***

There are 10 cancer hallmarks you will need to decide whether the article is related to, including:

activating invasion and metastasis

sustaining proliferative signaling

resisting cell death

cellular energetics

genomic instability and mutation

evading growth suppressors

inducing angiogenesis

enabling replicative immortality

avoiding immune destruction

tumor promoting inflammation

***OUTPUT***

The output should be in a json format, with relevant value for each class.

Put value 1 if the article is related to that class, 0 if the article is not related to that class. 

Please note one article can be related to multiple classes.

Example output:

{

  "activating invasion and metastasis": ,

  "sustaining proliferative signaling": ,

  "resisting cell death": ,

  "cellular energetics": ,

  "genomic instability and mutation": ,

  "evading growth suppressors": ,

  "inducing angiogenesis": ,

  "enabling replicative immortality": ,

  "avoiding immune destruction": ,

  "tumor promoting inflammation": 

}\n\n'''

        for i in range(0, shot):
            prompt_text += '***EXAMPLE-' + str(i+1) + '***\n\nINPUT: ' + ranked_examples[index][i][0] + '\n\nOUTPUT: \n\n' + ranked_examples[index][i][1] + '\n\n'

        prompt_text += '''INPUT: ''' + sentence + '\n\nOUTPUT: '

    elif task == 'litcovid':
        prompt_text = '''***TASK***

The task is to decide relevant COVID-19 topics of the article based on its abstract.

***INPUT***

The input is an abstract text.

***DOCUMENTATION***

There are 7 topics you will need to decide whether the article is related to. The followings are the topics and their definitions.

Mechanism: underlying cause(s) of COVID-19 infections and transmission and possible drug mechanism of action. 

Transmission: characteristics and modes of COVID-19 transmissions. 

Diagnosis: COVID-19 assessment through symptoms, test results and radiological features for COVID-19. 

Treatment: treatment strategies, therapeutic procedures and vaccine development for COVID-19. 

Prevention: prevention, control, mitigation and management strategies for COVID-19. 

Case Report: descriptions of specific patient cases related to COVID-19. 

Epidemic Forecasting: estimation on the trend of COVID-19 spread and related modeling approach. 

***OUTPUT***

The output should be in a json format, with relevant value for each topic: Treatment, Diagnosis, Prevention, Mechanism, Transmission, Epidemic Forecasting, Case Report.

Put value 1 if the article is related to that topic, 0 if the article is not related to that topic. 

Please note one article can be related to multiple topics.

Example output format:

{

  "Treatment": ,

  "Diagnosis": ,

  "Prevention": ,

  "Mechanism": ,

  "Transmission": ,

  "Epidemic Forecasting": ,

  "Case Report": 

}\n\n'''

        for i in range(0, shot):
            prompt_text += '***EXAMPLE-' + str(i+1) + '***\n\nINPUT: ' + ranked_examples[index][i][0] + '\n\nOUTPUT: \n\n' + ranked_examples[index][i][1] + '\n\n'

        prompt_text += '''INPUT: ''' + sentence + '\n\nOUTPUT: '

    return prompt_text

def run_ner_exp_os(in_gold_tsv_file,
                   in_gpt_tsv_file,
                   shot,
                   in_train_json,
                   in_similarity_json,
                   out_json_file,
                   task_name):

    ranked_examples = load_ner_examples(in_train_json, in_similarity_json)
    
    writer = open(out_json_file, 'w', encoding='utf-8')
    gold_tokens, gold_labels = [], []
    with open(in_gold_tsv_file, 'r', encoding='utf8') as reader:
        tokens, labels = [], []
        for line in reader:

            line = line.rstrip()

            if line == '':
                gold_tokens.append(tokens)
                gold_labels.append(labels)
                tokens, labels = [], []
                continue

            tks = line.split('\t')
            tokens.append(tks[0])
            labels.append(tks[1])

    gpt_tokens, gpt_labels, gpt_sentences  = load_tsv_to_token_label_list(in_gpt_tsv_file)
    gpt_ne_lists = convert_token_label_2_sent_ne_tuples_list(gpt_tokens, gpt_labels)
    gpt_labels = convert_ne_list_2_iob2(gold_tokens, gpt_ne_lists)

    sentences = load_conll_into_sentences(in_gold_tsv_file)

    out_dict = []
    tatal_num_token = 0
    print('=======>', out_json_file)
    print(len(sentences), len(gpt_labels), len(gpt_sentences), len(gpt_ne_lists))
    for index, ((sentence, label), pred_labels, response) in enumerate(zip(sentences, gpt_labels, gpt_sentences)):
        prompt_text = convert_2_prmopt(' '.join(sentence), 
                                       shot, 
                                       task_name, 
                                       ranked_examples, 
                                       index)
        _out_dict = {}
        _out_dict['index'] = index
        _out_dict['sentence'] = sentence
        _out_dict['gold'] = label
        _out_dict['pred'] = pred_labels
        _out_dict['prompt'] = prompt_text
        _out_dict['response'] = response
        out_dict.append(_out_dict)
    
    writer.write(json.dumps(out_dict, indent=4))
    writer.close()

def run_re_exp_os(in_gold_tsv_file,
                  in_gpt_tsv_file,
                  shot,
                  in_train_json,
                  in_similarity_json,
                  out_json_file,
                  task_name):
    
    ranked_examples = {}
    
    if task_name == 'hoc':
        ranked_examples = load_hoc_examples(in_train_json, in_similarity_json)
    elif task_name == 'litcovid':
        ranked_examples = load_litcoin_examples(in_train_json, in_similarity_json)
    elif task_name == 'chemprot':
        ranked_examples = load_re_examples(in_train_json, in_similarity_json)
    elif task_name == 'ddi':
        ranked_examples = load_re_examples(in_train_json, in_similarity_json)
    
    writer = open(out_json_file, 'w', encoding='utf-8')

    golds = []
    preds = []
    with open(in_gold_tsv_file, 'r', encoding='utf8') as reader:
        reader.readline()
        for line in reader:

            line = line.rstrip()

            if line == '':
                continue
            
            tks = line.split('\t')
            golds.append(tks)
    with open(in_gpt_tsv_file, 'r', encoding='utf8') as reader:
        reader.readline()
        for line in reader:

            line = line.rstrip()

            if line == '':
                continue
            
            tks = line.split('\t')
            preds.append(tks)

    out_dict = []
    tatal_num_token = 0
    print('=======>', out_json_file)
    print('=======>', len(golds), len(preds))
    print(len(golds[0]))
    for gold, pred in zip(golds, preds):

        (index, sentence, label) = gold
        (response, pred_label, _, _) = pred
        
        prompt_text = convert_2_prmopt(sentence, 
                                       shot, 
                                       task_name, 
                                       ranked_examples, 
                                       index)
        _out_dict = {}
        _out_dict['index'] = index
        _out_dict['sentence'] = sentence
        _out_dict['gold'] = label
        _out_dict['pred'] = pred_label
        _out_dict['prompt'] = prompt_text
        _out_dict['response'] = response
        out_dict.append(_out_dict)
    
    writer.write(json.dumps(out_dict, indent=4))
    writer.close()

if __name__ == '__main__':
    

    for gpt_version in ['gpt4', 'gpt35']:
        for shot in [1, 2, 5]:
            
            run_ner_exp_os(in_gold_tsv_file        = 'NCBI_Disease/datasets/full_set/test.tsv',
                                    in_gpt_tsv_file         = 'NCBI_Disease/test_' + gpt_version + '_t0_' + str(shot) +'s_pred.tsv',
                                    shot               = shot,
                                    in_train_json      = 'data_Early/NCBI/NCBI_Disease_train.json',
                                    in_similarity_json = 'data_Early/NCBI/similarity.rank.json',
                                    out_json_file      = 'NCBI_Disease_' + gpt_version + '_' + str(shot) +'s.json',
                                    task_name='ncbi_disease')
            
            run_ner_exp_os(in_gold_tsv_file         = 'BC5CDR_Chemical/datasets/full_set/test.tsv',
                                    in_gpt_tsv_file          = 'BC5CDR_Chemical/test_' + gpt_version + '_t0_' + str(shot) +'s_pred.tsv',
                                    shot                = shot,
                                    in_train_json       = 'data_Early/BC5CDR/BC5CDR_Chemical_train.json',
                                    in_similarity_json  = 'data_Early/BC5CDR/similarity.rank.json',
                                    out_json_file      = 'BC5CDR_Chemical_' + gpt_version + '_' + str(shot) +'s.json',
                                    task_name='bc5cdr_chemical')
                
            run_re_exp_os(in_gold_tsv_file        = 'Hoc/datasets/full_set/test.tsv',
                        in_gpt_tsv_file         = 'Hoc/test_' + gpt_version + '_t0_' + str(shot) +'s_pred.tsv',
                        shot               = shot,
                        in_train_json      = 'data_Early/Hoc/Hoc_train.json',
                        in_similarity_json = 'data_Early/Hoc/similarity.rank.json',
                        out_json_file      = 'Hoc_' + gpt_version + '_' + str(shot) +'s.json',
                        task_name='hoc')

            run_re_exp_os(in_gold_tsv_file        = 'LitCovid/datasets/full_set/test.tsv',
                                in_gpt_tsv_file         = 'LitCovid/test_' + gpt_version + '_t0_' + str(shot) +'s_pred.tsv',
                                shot               = shot,
                                in_train_json      = 'data_Early/LitCovid/LitCovid_train.json',
                                in_similarity_json = 'data_Early/LitCovid/similarity.rank.json',
                                out_json_file      = 'LitCovid_' + gpt_version + '_' + str(shot) +'s.json',
                                task_name='litcovid')
        
            run_re_exp_os(in_gold_tsv_file        = 'Chemprot/datasets/full_set/test.tsv',
                                in_gpt_tsv_file         = 'Chemprot/test_' + gpt_version + '_t0_' + str(shot) +'s_pred.tsv',
                                shot               = shot,
                                in_train_json      = 'data_Early/Chemprot/Chemprot_train.json',
                                in_similarity_json = 'data_Early/Chemprot/similarity.rank.json',
                                out_json_file      = 'Chemprot_' + gpt_version + '_' + str(shot) +'s.json',
                                task_name='chemprot')
            
            run_re_exp_os(in_gold_tsv_file        = 'DDI/datasets/full_set/test.tsv',
                        in_gpt_tsv_file         = 'DDI/test_' + gpt_version + '_t0_' + str(shot) +'s_pred.tsv',
                        shot               = shot,
                        in_train_json      = 'data_Early/DDI/DDI_train.json',
                        in_similarity_json = 'data_Early/DDI/similarity.rank.json',
                        out_json_file      = 'DDI_' + gpt_version + '_' + str(shot) +'s.json',
                        task_name='ddi')