from collections import defaultdict
import openai
import time
import re
import json
import tiktoken
    
def load_conll_into_sentences(in_tsv_file):

    with open(in_tsv_file, 'r', encoding='utf-8') as in_tsv_reader:
        sentences = []
        sentence = []
        for line in in_tsv_reader:
            line = line.rstrip()
            if line == '':
                if len(sentence) > 0:
                    sentences.append((str(len(sentence)), ' '.join(sentence)))
                    sentence = []
            else:
                sentence.append(line.split('\t')[0])
        if len(sentence) > 0:
            sentences.append((str(len(sentence)), sentence))
    
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

def __set_opengpt(gpt_version):

    if gpt_version == 'gpt-4':
        openai.api_type = "azure"
        openai.api_version = "2023-03-15-preview"
        openai.api_base = "YOUT_API_BASE"
        openai.api_key = "YOUT_API_KEY" 

        engine            = 'gpt-4'
        waiting_time      = 2
    elif gpt_version == 'gpt-3.5' or gpt_version == 'gpt-3.5-turbo':
        
        openai.api_type = "azure"
        openai.api_version = "2023-03-15-preview"
        openai.api_base = "YOUT_API_BASE"
        openai.api_key = "YOUT_API_KEY"
        engine            = 'gpt-35-turbo'
        waiting_time      = 2

    return engine, waiting_time
    

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    if encoding_name == 'gpt-3.5':
        encoding_name = 'gpt-3.5-turbo'
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

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

def run_ncbi_disease_exp_os(in_tsv_file,
                            out_tsv_file,
                            shot,
                            in_train_json,
                            in_similarity_json,
                            gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_conll_into_sentences(in_tsv_file)

    ranked_examples = load_ner_examples(in_train_json, in_similarity_json)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    tatal_num_token = 0

    for (index, sentence) in sentences:

        prompt_text = '''TASK: the task is to extract disease entities in a sentence.

INPUT: the input is a sentence.

OUTPUT: the output is an HTML that highlights all the disease entities with <span> ... </span> in the sentence.

The following example format is provided:\n\n'''

        if index in ranked_examples:
            for i in range(0, shot):
                prompt_text += 'Example-' + str(i+1) + ' Sentence: ' + ranked_examples[index][i][0] + '\n\nHTML: ' + ranked_examples[index][i][1] + '\n\n'

        prompt_text += '''Sentence: ''' + sentence + '\n\nHTML: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
        
        num_token = num_tokens_from_string(prompt_text, gpt_version)
        try:
            # model="gpt-3.5-turbo",  # Replace with the desired chat model
            response = openai.ChatCompletion.create(
                engine      = engine,  # Replace with the desired chat model
                messages    = historys,
                n           = 1,
                stop        = None,
                temperature = 0.,
            )
            
            message = response.choices[0].message
            print(message['content'])
            num_out_token = num_tokens_from_string(message['content'], gpt_version)
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\t' + str(num_token) + '\t'+ str(num_out_token) +'\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\t' + str(num_token) + '\t0\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)

    out_tsv_writer.close()

def run_bc5cdr_chemical_exp_os(in_tsv_file,
                               out_tsv_file,
                               shot,
                               in_train_json,
                               in_similarity_json,
                               gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_conll_into_sentences(in_tsv_file)

    ranked_examples = load_ner_examples(in_train_json, in_similarity_json)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    tatal_num_token = 0

    for (index, sentence) in sentences:

        prompt_text = '''TASK: the task is to extract chemical entities in a sentence.

INPUT: the input is a sentence.

OUTPUT: the output is an HTML that highlights all the chemical entities with <span> ... </span> in the sentence.

The following example format is provided:\n\n'''

        if index in ranked_examples:
            for i in range(0, shot):
                prompt_text += 'Example-' + str(i+1) + ' Sentence: ' + ranked_examples[index][i][0] + '\n\nHTML: ' + ranked_examples[index][i][1] + '\n\n'

        prompt_text += '''Sentence: ''' + sentence + '\n\nHTML: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
        num_token = num_tokens_from_string(prompt_text, gpt_version)
        try:
            # model="gpt-3.5-turbo",  # Replace with the desired chat model
            response = openai.ChatCompletion.create(
                engine      = engine,  # Replace with the desired chat model
                messages    = historys,
                n           = 1,
                stop        = None,
                temperature = 0.,
            )
            
            message = response.choices[0].message
            print(message['content'])
            num_out_token = num_tokens_from_string(message['content'], gpt_version)
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\t' + str(num_token) + '\t'+ str(num_out_token) +'\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\t' + str(num_token) + '\t0\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

def run_chemprot_exp_os(in_tsv_file,
                        out_tsv_file,
                        shot,
                        in_train_json,
                        in_similarity_json,
                        gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_relation_into_sentences(in_tsv_file)

    ranked_examples = load_re_examples(in_train_json, in_similarity_json)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    tatal_num_token = 0

    for (index, sentence) in sentences:

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

        if index in ranked_examples:
            for i in range(0, shot):
                prompt_text += 'Example-' + str(i+1) + ' Q: ' + ranked_examples[index][i][0] + '\n\nA: ' + ranked_examples[index][i][1] + '\n\n'

        prompt_text += '''Q: ''' + sentence + '\n\nA: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
        num_token = num_tokens_from_string(prompt_text, gpt_version)
        try:
            # model="gpt-3.5-turbo",  # Replace with the desired chat model
            response = openai.ChatCompletion.create(
                engine      = engine,  # Replace with the desired chat model
                messages    = historys,
                n           = 1,
                stop        = None,
                temperature = 0.,
            )
            
            message = response.choices[0].message
            print(message['content'])
            num_out_token = num_tokens_from_string(message['content'], gpt_version)
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\t' + str(num_token) + '\t'+ str(num_out_token) +'\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\t' + str(num_token) + '\t0\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

def run_ddi_exp_os(in_tsv_file,
                   out_tsv_file,
                   shot,
                   in_train_json,
                   in_similarity_json,
                   gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_relation_into_sentences(in_tsv_file)

    ranked_examples = load_re_examples(in_train_json, in_similarity_json)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    tatal_num_token = 0

    for (index, sentence) in sentences:

        prompt_text = '''TASK: the task is to classify relations between two drugs for a sentence. 

INPUT: the input is a sentence where the drugs are labeled as @DRUG$. 

OUTPUT: your task is to select one out of the five types of relations ('DDI-effect', 'DDI-mechanism', 'DDI-advise', 'DDI-false', and 'DDI-int') for the drugs without any explanation or other characters:

DDI-mechanism: This type is used to annotate DDIs that are described by their PK mechanism (e.g. Grepafloxacin may inhibit the metabolism of theobromine)

DDI-effect: This type is used to annotate DDIs describing an effect (e.g. In uninfected volunteers, 46% developed rash while receiving SUSTIVA and clarithromycin) or a PD mechanism (e.g. Chlorthalidone may potentiate the action of other antihypertensive drugs)

DDI-advise: This type is used when a recommendation or advice regarding a drug interaction is given (e.g. UROXATRAL should not be used in combination with other alpha-blockers)

DDI-int: This type is used when a DDI appears in the text without providing any additional information (e.g. The interaction of omeprazole and ketoconazole has been established)

DDI-false, This type is used when no DDI relation appears

The following examples are provided:\n\n'''

        if index in ranked_examples:
            for i in range(0, shot):
                prompt_text += 'Example-' + str(i+1) + ' Q: ' + ranked_examples[index][i][0] + '\n\nA: ' + ranked_examples[index][i][1] + '\n\n'

        prompt_text += '''Q: ''' + sentence + '\n\nA: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
        num_token = num_tokens_from_string(prompt_text, gpt_version)
        try:
            # model="gpt-3.5-turbo",  # Replace with the desired chat model
            response = openai.ChatCompletion.create(
                engine      = engine,  # Replace with the desired chat model
                messages    = historys,
                n           = 1,
                stop        = None,
                temperature = 0.,
            )
            
            message = response.choices[0].message
            print(message['content'])
            num_out_token = num_tokens_from_string(message['content'], gpt_version)
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\t' + str(num_token) + '\t'+ str(num_out_token) +'\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\t' + str(num_token) + '\t0\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)

    out_tsv_writer.close()

def run_hoc_exp_os(in_tsv_file,
                   out_tsv_file,
                   shot,
                   in_train_json,
                   in_similarity_json,
                   gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_relation_into_sentences(in_tsv_file)

    ranked_examples = load_hoc_examples(in_train_json, in_similarity_json)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    tatal_num_token = 0

    for (index, sentence) in sentences:

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

        if index in ranked_examples:
            for i in range(0, shot):
                prompt_text += '***EXAMPLE-' + str(i+1) + '***\n\nINPUT: ' + ranked_examples[index][i][0] + '\n\nOUTPUT: \n\n' + ranked_examples[index][i][1] + '\n\n'

        prompt_text += '''INPUT: ''' + sentence + '\n\nOUTPUT: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
        num_token = num_tokens_from_string(prompt_text, gpt_version)
        try:
            # model="gpt-3.5-turbo",  # Replace with the desired chat model
            response = openai.ChatCompletion.create(
                engine      = engine,  # Replace with the desired chat model
                messages    = historys,
                n           = 1,
                stop        = None,
                temperature = 0.,
            )
            
            message = response.choices[0].message
            print(message['content'])
            num_out_token = num_tokens_from_string(message['content'], gpt_version)
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\t' + str(num_token) + '\t'+ str(num_out_token) +'\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\t' + str(num_token) + '\t0\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

def run_litcovid_exp_os(in_tsv_file,
                        out_tsv_file,
                        shot,
                        in_train_json,
                        in_similarity_json,
                        gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_relation_into_sentences(in_tsv_file)

    ranked_examples = load_litcoin_examples(in_train_json, in_similarity_json)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    tatal_num_token = 0

    for (index, sentence) in sentences:

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

        if index in ranked_examples:
            for i in range(0, shot):
                prompt_text += '***EXAMPLE-' + str(i+1) + '***\n\nINPUT: ' + ranked_examples[index][i][0] + '\n\nOUTPUT: \n\n' + ranked_examples[index][i][1] + '\n\n'

        prompt_text += '''INPUT: ''' + sentence + '\n\nOUTPUT: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
        num_token = num_tokens_from_string(prompt_text, gpt_version)
        try:
            # model="gpt-3.5-turbo",  # Replace with the desired chat model
            response = openai.ChatCompletion.create(
                engine      = engine,  # Replace with the desired chat model
                messages    = historys,
                n           = 1,
                stop        = None,
                temperature = 0.,
            )
            
            message = response.choices[0].message
            print(message['content'])
            num_out_token = num_tokens_from_string(message['content'], gpt_version)
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\t' + str(num_token) + '\t'+ str(num_out_token) +'\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\t' + str(num_token) + '\t0\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

if __name__ == '__main__':
    
    for gpt_version in ['gpt-4', 'gpt-3.5']

        for shot in [1, 2, 5]:
            
            run_ncbi_disease_exp_os(in_tsv_file        = 'NCBI_Disease/datasets/full_set/test.tsv',
                                    out_tsv_file       = 'NCBI_Disease/test_gpt4_t0_' + str(shot) +'s_pred.tsv',
                                    shot               = shot,
                                    in_train_json      = 'data_Early/NCBI/NCBI_Disease_train.json',
                                    in_similarity_json = 'data_Early/NCBI/similarity.rank.json',
                                    gpt_version        = gpt_version)
                
            run_bc5cdr_chemical_exp_os(in_tsv_file         = 'BC5CDR_Chemical/datasets/full_set/test.tsv',
                                       out_tsv_file        = 'BC5CDR_Chemical/test_gpt4_t0_' + str(shot) +'s_pred.tsv',
                                       shot                = shot,
                                       in_train_json       = 'data_Early/BC5CDR/BC5CDR_Chemical_train.json',
                                       in_similarity_json  = 'data_Early/BC5CDR/similarity.rank.json',
                                       gpt_version         = gpt_version)
                
            run_hoc_exp_os(in_tsv_file        = 'Hoc/datasets/full_set/test.tsv',
                           out_tsv_file       = 'Hoc/test_gpt4_t0_' + str(shot) +'s_pred.tsv',
                           shot               = shot,
                           in_train_json      = 'data_Early/Hoc/Hoc_train.json',
                           in_similarity_json = 'data_Early/Hoc/similarity.rank.json',
                           gpt_version        = gpt_version)

            run_litcovid_exp_os(in_tsv_file        = 'LitCovid/datasets/full_set/test.tsv',
                                out_tsv_file       = 'LitCovid/test_gpt4_t0_' + str(shot) +'s_pred.tsv',
                                shot               = shot,
                                in_train_json      = 'data_Early/LitCovid/LitCovid_train.json',
                                in_similarity_json = 'data_Early/LitCovid/similarity.rank.json',
                                gpt_version        = gpt_version)
        
            run_chemprot_exp_os(in_tsv_file        = 'Chemprot/datasets/full_set/test.tsv',
                                out_tsv_file       = 'Chemprot/test_gpt4_t0_' + str(shot) +'s_pred.tsv',
                                shot               = shot,
                                in_train_json      = 'data_Early/Chemprot/Chemprot_train.json',
                                in_similarity_json = 'data_Early/Chemprot/similarity.rank.json',
                                gpt_version        = gpt_version)
            
            run_ddi_exp_os(in_tsv_file        = 'DDI/datasets/full_set/test.tsv',
                           out_tsv_file       = 'DDI/test_gpt4_t0_' + str(shot) +'s_pred.tsv',
                           shot               = shot,
                           in_train_json      = 'data_Early/DDI/DDI_train.json',
                           in_similarity_json = 'data_Early/DDI/similarity.rank.json',
                           gpt_version        = gpt_version)