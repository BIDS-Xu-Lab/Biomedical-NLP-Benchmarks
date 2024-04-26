from collections import defaultdict
import openai
import time
import re

    
def load_conll_into_sentences(in_tsv_file):

    with open(in_tsv_file, 'r', encoding='utf-8') as in_tsv_reader:
        sentences = []
        sentence = []
        for line in in_tsv_reader:
            line = line.rstrip()
            if line == '':
                if len(sentence) > 0:
                    sentences.append(' '.join(sentence))
                    sentence = []
            else:
                sentence.append(line.split('\t')[0])
        if len(sentence) > 0:
            sentences.append(sentence)
    
    return sentences

def load_relation_into_sentences(in_tsv_file):

    with open(in_tsv_file, 'r', encoding='utf-8') as in_tsv_reader:
        sentences = []

        for line in in_tsv_reader:

            tks = line.rstrip().split('\t')
            if len(tks) == 3:
                sentence = tks[1]
                sentences.append(sentence)

    return sentences

def __set_opengpt(gpt_version):

    if gpt_version == 'gpt-4':
        openai.api_type = "azure"
        openai.api_version = "2023-03-15-preview"
        openai.api_base = "<YOUR_API_BASE>"

        openai.api_key = "<YOUR_API_KEY>" 

        engine            = 'gpt-4'
        waiting_time      = 7
    elif gpt_version == 'gpt-3.5' or gpt_version == 'gpt-3.5-turbo':
        
        openai.api_type = "azure"
        openai.api_version = "2023-03-15-preview"
        openai.api_base = "<YOUR_API_BASE>"
        openai.api_key = "<YOUR_API_KEY>"
        engine            = 'gpt-35-turbo'
        waiting_time      = 2

    return engine, waiting_time
    

def run_ncbi_disease_exp_os(in_tsv_file,
                         out_tsv_file,
                         gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_conll_into_sentences(in_tsv_file)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    for sentence in sentences:

        prompt_text = '''TASK: the task is to extract disease entities in a sentence.

INPUT: the input is a sentence.

OUTPUT: the output is an HTML that highlights all the disease entities with <span> ... </span> in the sentence.

The following example format is provided:

Example-1 Sentence: A common MSH2 mutation in English and North American HNPCC families : origin , phenotypic expression , and sex specific differences in colorectal cancer .

Example-1 HTML: A common MSH2 mutation in English and North American <span> HNPCC </span> families : origin , phenotypic expression , and sex specific differences in <span> colorectal cancer </span> .

Example-1 Sentence: OBJECTIVES : The United Kingdom Parkinson ' s Disease Research Group ( UKPDRG ) trial found an increased mortality in patients with Parkinson ' s disease ( PD ) randomized to receive 10 mg selegiline per day and L - dopa compared with those taking L - dopa alone .

Example-1 HTML: OBJECTIVES : The United Kingdom Parkinson ' s Disease Research Group ( UKPDRG ) trial found an increased mortality in patients with Parkinson ' s disease ( PD ) randomized to receive 10 mg <span> selegiline </span> per day and <span> L - dopa </span> compared with those taking <span> L - dopa </span> alone .



Sentence: ''' + sentence + '\n\nHTML: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
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
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

def run_bc5cdr_chemical_exp_os(in_tsv_file,
                         out_tsv_file,
                         gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_conll_into_sentences(in_tsv_file)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    for sentence in sentences:

        prompt_text = '''TASK: the task is to extract chemical entities in a sentence.

INPUT: the input is a sentence.

OUTPUT: the output is an HTML that highlights all the chemical entities with <span> ... </span> in the sentence.

The following example format is provided:

Example-1 Sentence: OBJECTIVES : The United Kingdom Parkinson ' s Disease Research Group ( UKPDRG ) trial found an increased mortality in patients with Parkinson ' s disease ( PD ) randomized to receive 10 mg selegiline per day and L - dopa compared with those taking L - dopa alone .

Example-1 HTML: OBJECTIVES : The United Kingdom Parkinson ' s Disease Research Group ( UKPDRG ) trial found an increased mortality in patients with Parkinson ' s disease ( PD ) randomized to receive 10 mg <span> selegiline </span> per day and <span> L - dopa </span> compared with those taking <span> L - dopa </span> alone .


Sentence: ''' + sentence + '\n\nHTML: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
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
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

def run_chemprot_exp_zs(in_tsv_file,
                         out_tsv_file,
                         gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_relation_into_sentences(in_tsv_file)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    for sentence in sentences:

        prompt_text = '''TASK: the task is to classify relations between a chemical and a gene for a sentence. 

INPUT: the input is a sentence where the chemical is labeled as @CHEMICAL$ and the gene is labeled as @GENE$ accordingly in a sentence. 

OUTPUT: your task is to select one out of the six types of relations ('CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9', and 'false') for the gene and chemical without any explanation or other characters:

CPR:3, which includes UPREGULATOR, ACTIVATOR, and INDIRECT UPREGULATOR

CPR:4, which includes DOWNREGULATOR, INHIBITOR ,and INDIRECT DOWNREGULATOR

CPR:5, which includes AGONIST, AGONIST ACTIVATOR, and AGONIST INHIBITOR

CPR:6, which includes ANTAGONIST

CPR:9, which includes SUBSTRATE, PRODUCT OF and SUBSTRATE PRODUCT OF

false, which indicates no relations

The following example formats are provided:

Example-1 Q: XXXX .... XXXX.

Example-1 A: CPR:3

Example-2 Q: XXXX .... XXXX.

Example-2 A: false

Q: ''' + sentence + '\n\nA: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
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
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

def run_ddi_exp_zs(in_tsv_file,
                         out_tsv_file,
                         gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_relation_into_sentences(in_tsv_file)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    for sentence in sentences:

        prompt_text = '''TASK: the task is to classify relations between two drugs for a sentence. 

INPUT: the input is a sentence where the drugs are labeled as @DRUG$. 

OUTPUT: your task is to select one out of the five types of relations ('DDI-effect', 'DDI-mechanism', 'DDI-advise', 'DDI-false', and 'DDI-int') for the drugs without any explanation or other characters:

DDI-mechanism: This type is used to annotate DDIs that are described by their PK mechanism (e.g. Grepafloxacin may inhibit the metabolism of theobromine)

DDI-effect: This type is used to annotate DDIs describing an effect (e.g. In uninfected volunteers, 46% developed rash while receiving SUSTIVA and clarithromycin) or a PD mechanism (e.g. Chlorthalidone may potentiate the action of other antihypertensive drugs)

DDI-advise: This type is used when a recommendation or advice regarding a drug interaction is given (e.g. UROXATRAL should not be used in combination with other alpha-blockers)

DDI-int: This type is used when a DDI appears in the text without providing any additional information (e.g. The interaction of omeprazole and ketoconazole has been established)

DDI-false, This type is used when no DDI relation appears


The following example formats are provided:

Example-1 Q: XXXX .... XXXX.

Example-1 A: DDI-effect

Example-2 Q: XXXX .... XXXX.

Example-2 A: DDI-false

Q: ''' + sentence + '\n\nA: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
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
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

def run_hoc_exp_zs(in_tsv_file,
                         out_tsv_file,
                         gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_relation_into_sentences(in_tsv_file)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    for sentence in sentences:

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

}

***EXAMPLES***

INPUT: XXXX .... XXXX.

OUTPUT:

{

  "activating invasion and metastasis": 0,

  "sustaining proliferative signaling": 1,

  "resisting cell death": 1,

  "cellular energetics": 0,

  "genomic instability and mutation": 0,

  "evading growth suppressors": 0,

  "inducing angiogenesis": 0,

  "enabling replicative immortality": 0,

  "avoiding immune destruction": 0,

  "tumor promoting inflammation": 0

}

INPUT: ''' + sentence + '\n\nOUTPUT: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
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
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

def run_litcovid_exp_zs(in_tsv_file,
                         out_tsv_file,
                         gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_relation_into_sentences(in_tsv_file)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    for sentence in sentences:

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

}

***EXAMPLES***

INPUT: XXXX .... XXXX.

OUTPUT:

{

  "Treatment": 1,

  "Diagnosis": 0,

  "Prevention": 0,

  "Mechanism": 0,

  "Transmission": 0,

  "Epidemic Forecasting": 0,

  "Case Report": 0

}

INPUT: ''' + sentence + '\n\nOUTPUT: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
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
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

def run_ncbi_disease_exp_zs(in_tsv_file,
                         out_tsv_file,
                         gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_conll_into_sentences(in_tsv_file)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    for sentence in sentences:

        prompt_text = '''TASK: the task is to extract disease entities in a sentence.

INPUT: the input is a sentence.

OUTPUT: the output is an HTML that highlights all the disease entities with <span> ... </span> in the sentence.

Sentence: ''' + sentence + '\n\nHTML: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
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
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

def run_bc5cdr_chemical_exp_zs(in_tsv_file,
                         out_tsv_file,
                         gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_conll_into_sentences(in_tsv_file)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    for sentence in sentences:

        prompt_text = '''TASK: the task is to extract chemical entities in a sentence.

INPUT: the input is a sentence.

OUTPUT: the output is an HTML that highlights all the chemical entities with <span> ... </span> in the sentence.

Sentence: ''' + sentence + '\n\nHTML: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
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
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

def run_chemprot_exp_os(in_tsv_file,
                         out_tsv_file,
                         gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_relation_into_sentences(in_tsv_file)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    for sentence in sentences:

        prompt_text = '''TASK: the task is to classify relations between a chemical and a gene for a sentence. 

INPUT: the input is a sentence where the chemical is labeled as @CHEMICAL$ and the gene is labeled as @GENE$ accordingly in a sentence. 

OUTPUT: your task is to select one out of the six types of relations ('CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9', and 'false') for the gene and chemical without any explanation or other characters:

CPR:3, which includes UPREGULATOR, ACTIVATOR, and INDIRECT UPREGULATOR

CPR:4, which includes DOWNREGULATOR, INHIBITOR ,and INDIRECT DOWNREGULATOR

CPR:5, which includes AGONIST, AGONIST ACTIVATOR, and AGONIST INHIBITOR

CPR:6, which includes ANTAGONIST

CPR:9, which includes SUBSTRATE, PRODUCT OF and SUBSTRATE PRODUCT OF

false, which indicates no relations

The following examples are provided:

Example-1 Q: Three lines of evidence suggest that calmodulin inhibition is not responsible for the inhibition of binding and endocytosis: 1) Promethazine, a phenothiazine that is a poor inhibitor of calmodulin, is nearly as effective as TFP at inhibiting endocytosis; calmidazolium, a potent inhibitor of several calmodulin functions, did not cause a loss of binding; 2) the microinjection of calmodulin into cells did not reverse the effects of @CHEMICAL$; using pressure microinjection, we introduced up to a 100-fold excess of @GENE$ over native levels into individual gerbil fibroma cells; using rhodamine-labeled alpha 2-macroglobulin, we saw that the W-7 induced inhibition of receptor-mediated endocytosis was the same in injected and uninjected cells; 3) we injected calcineurin, a calmodulin-binding protein, into cells (1-3 pg/cell) and observed no effect on the receptor-mediated endocytosis of rhodamine-labeled alpha 2-macroglobulin.

Example-1 A: false

Q: ''' + sentence + '\n\nA: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
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
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

def run_ddi_exp_os(in_tsv_file,
                         out_tsv_file,
                         gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_relation_into_sentences(in_tsv_file)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    for sentence in sentences:

        prompt_text = '''TASK: the task is to classify relations between two drugs for a sentence. 

INPUT: the input is a sentence where the drugs are labeled as @DRUG$. 

OUTPUT: your task is to select one out of the five types of relations ('DDI-effect', 'DDI-mechanism', 'DDI-advise', 'DDI-false', and 'DDI-int') for the drugs without any explanation or other characters:

DDI-mechanism: This type is used to annotate DDIs that are described by their PK mechanism (e.g. Grepafloxacin may inhibit the metabolism of theobromine)

DDI-effect: This type is used to annotate DDIs describing an effect (e.g. In uninfected volunteers, 46% developed rash while receiving SUSTIVA and clarithromycin) or a PD mechanism (e.g. Chlorthalidone may potentiate the action of other antihypertensive drugs)

DDI-advise: This type is used when a recommendation or advice regarding a drug interaction is given (e.g. UROXATRAL should not be used in combination with other alpha-blockers)

DDI-int: This type is used when a DDI appears in the text without providing any additional information (e.g. The interaction of omeprazole and ketoconazole has been established)

DDI-false, This type is used when no DDI relation appears

The following examples are provided:

Example-1 Q: Interaction with Other @DRUG$: MEPERIDINE SHOULD BE USED WITH GREAT CAUTION AND IN REDUCED DOSAGE IN PATIENTS WHO ARE CONCURRENTLY RECEIVING OTHER NARCOTIC ANALGESICS, GENERAL @DRUG$, PHENOTHIAZINES, OTHER TRANQUILIZERS, SEDATIVE-HYPNOTICS (INCLUDING BARBITURATES), TRICYCLIC ANTIDEPRESSANTS AND OTHER CNS DEPRESSANTS (INCLUDING ALCOHOL).

Example-1 A: DDI-false

Q: ''' + sentence + '\n\nA: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
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
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

def run_hoc_exp_os(in_tsv_file,
                         out_tsv_file,
                         gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_relation_into_sentences(in_tsv_file)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    for sentence in sentences:

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

}

***EXAMPLES***

INPUT: Breast cancer is the most commonly diagnosed cancer in women in the world and is one of the 

leading causes of death due to cancer . Health benefits have been linked to additive and synergistic combinations 

of phytochemicals in fruits and vegetables . Nigella sativa has been shown to possess anti-carcinogenic activity , 

inhibiting growth of several cancer cell lines in vitro . However , the molecular mechanisms of the anti-cancer 

properties of Nigella sativa phytochemical extracts have not been completely understood . Our data showed that 

Nigella sativa extracts significantly inhibited human breast cancer MDA-MB-231 cell proliferation at doses of 2.5-5 Î¼g/mL ( P&lt;0.05 ) . 

Apoptotic induction in MDA-MB-231 cells was observed in a dose-dependent manner after exposure to Nigella sativa 

extracts for 48 h . Real time PCR and flow cytometry analyses suggested that Nigella sativa extracts possess 

the ability to suppress the proliferation of human breast cancer cells through induction of apoptosis.

OUTPUT:

{

  "activating invasion and metastasis": 0,

  "sustaining proliferative signaling": 1,

  "resisting cell death": 1,

  "cellular energetics": 0,

  "genomic instability and mutation": 0,

  "evading growth suppressors": 0,

  "inducing angiogenesis": 0,

  "enabling replicative immortality": 0,

  "avoiding immune destruction": 0,

  "tumor promoting inflammation": 0

}

INPUT: ''' + sentence + '\n\nOUTPUT: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
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
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

def run_litcovid_exp_os(in_tsv_file,
                         out_tsv_file,
                         gpt_version = 'gpt-4'):
    
    engine, waiting_time = __set_opengpt(gpt_version)
    
    sentences = load_relation_into_sentences(in_tsv_file)

    out_tsv_writer = open(out_tsv_file, 'w', encoding='utf-8')

    for sentence in sentences:

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

}

***EXAMPLES***

INPUT: COVID-19 and implications for dermatological and allergological diseases. COVID-19, 

caused by the coronavirus SARS-CoV-2, has become pandemic. A further level of complexity opens up as 

soon as we look at diseases whose pathogenesis and therapy involve different immunological signaling 

pathways, which are potentially affected by COVID-19. Medical treatments must often be reassessed 

and questioned in connection with this infection. This article summarizes the current knowledge of 

COVID-19 in the light of major dermatological and allergological diseases. It identifies medical areas 

lacking sufficient data and draws conclusions for the management of our patients during the pandemic. 

We focus on common chronic inflammatory skin diseases with complex immunological pathogenesis: 

psoriasis, eczema including atopic dermatitis, type I allergies, autoimmune blistering and 

inflammatory connective tissue diseases, vasculitis, and skin cancers. Since several other 

inflammatory skin diseases display related or comparable immunological reactions, clustering of 

the various inflammatory dermatoses into different disease patterns may help with therapeutic decisions. 

Thus, following these patterns of skin inflammation, our review may supply treatment recommendations and 

thoughtful considerations for disease management even beyond the most frequent diseases discussed here.

OUTPUT:

{

  "Treatment": 1,

  "Diagnosis": 0,

  "Prevention": 0,

  "Mechanism": 0,

  "Transmission": 0,

  "Epidemic Forecasting": 0,

  "Case Report": 0

}

INPUT: ''' + sentence + '\n\nOUTPUT: '
        
        print(prompt_text)
        historys = []
        historys.append({"role": "user", "content": prompt_text})
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
            out_tsv_writer.write(sentence + '\t' + message['content'].replace('\n', ' ').rstrip() + '\n')
            out_tsv_writer.flush()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print(sentence + ' failed!')
            out_tsv_writer.write(sentence + '\t' + '' + '\n')
            out_tsv_writer.flush()
        time.sleep(waiting_time)
    out_tsv_writer.close()

if __name__ == '__main__':
    
    for gpt_version in ['gpt-4', 'gpt-3.5']:
    
        run_ncbi_disease_exp_os(in_tsv_file  = 'NCBI_Disease/datasets/full_set/test.tsv',
                            out_tsv_file = 'NCBI_Disease/test_gpt4_t0_os_pred.tsv',
                            gpt_version  = gpt_version)
        
        
        run_bc5cdr_chemical_exp_zs(in_tsv_file  = 'BC5CDR_Chemical/datasets/full_set/test.tsv',
                                out_tsv_file = 'BC5CDR_Chemical/test_gpt4_t0_zs_pred.tsv',
                                gpt_version  = gpt_version)
        
        run_bc5cdr_chemical_exp_os(in_tsv_file  = 'BC5CDR_Chemical/datasets/full_set/test.tsv',
                                out_tsv_file = 'BC5CDR_Chemical/test_gpt4_t0_os_pred.tsv',
                                gpt_version  = gpt_version)
            
        run_hoc_exp_os(in_tsv_file  = 'Hoc/datasets/full_set/test.tsv',
                    out_tsv_file = 'Hoc/test_gpt4_t0_os_pred.tsv',
                    gpt_version  = gpt_version)
        
        run_litcovid_exp_os(in_tsv_file  = 'LitCovid/datasets/full_set/test.tsv',
                        out_tsv_file = 'LitCovid/test_gpt4_t0_os_pred.tsv',
                        gpt_version  = gpt_version)
        
        run_chemprot_exp_zs(in_tsv_file  = 'Chemprot/datasets/full_set/test.tsv',
                        out_tsv_file = 'Chemprot/test_gpt4_t0_zs_pred.tsv',
                        gpt_version  = gpt_version)
        
        run_ddi_exp_zs(in_tsv_file  = 'DDI/datasets/full_set/test.tsv',
                    out_tsv_file = 'DDI/test_gpt4_t0_zs_pred.tsv',
                    gpt_version  = gpt_version)
        
        run_hoc_exp_zs(in_tsv_file  = 'Hoc/datasets/full_set/test.tsv',
                    out_tsv_file = 'Hoc/test_gpt4_t0_zs_pred.tsv',
                    gpt_version  = gpt_version)
        
        run_litcovid_exp_zs(in_tsv_file  = 'LitCovid/datasets/full_set/test.tsv',
                        out_tsv_file = 'LitCovid/test_gpt4_t0_zs_pred.tsv',
                        gpt_version  = gpt_version)
        
        
        run_chemprot_exp_os(in_tsv_file  = 'Chemprot/datasets/full_set/test.tsv',
                        out_tsv_file = 'Chemprot/test_gpt4_t0_os_pred.tsv',
                        gpt_version  = gpt_version)
        
        run_ddi_exp_os(in_tsv_file  = 'DDI/datasets/full_set/test.tsv',
                    out_tsv_file = 'DDI/test_gpt4_t0_os_pred.tsv',
                    gpt_version  = gpt_version)
        
        
        run_ncbi_disease_exp_zs(in_tsv_file  = 'NCBI_Disease/datasets/full_set/test.tsv',
                            out_tsv_file = 'NCBI_Disease/test_gpt4_t0_zs_pred.tsv',
                            gpt_version  = gpt_version)
        