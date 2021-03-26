"""
CREATED BY: Gianna Maniaci
LAST UPDATED: July 15, 2020.
PURPOSE: Creates a new JSON file containing the specified data (there is a spot in make_json_model_and_data() where
you specify which columns you want to keep from the data) and the new model, whichever type was specified when calling the
program ('bow', 'tfidf', 'lda', 'doc2vec'). This can then be fed into a clustering algorithm. The models used are also saved. 
"""

import argparse
import json
import spacy
import en_core_web_sm
import re
import gensim
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def main(train_file, subset, preprocess, representation, comcast_data_type):
    """
    Args:
      - train_file (str) file location of transcripts
      - subset (bool) if true, keep only first 100 and last 100
          words of transcript
      - preprocess (bool) whether to expand contractions, remove stopwords,
          and lemmatize words
      - representation (str: {'bow', 'tfidf', 'lda', 'doc2vec'})
      - comcast_data_type (str: {'res', 'bus'}). string stating whether Comcast Business or Residential
          data is being used
    Return:
      - model_representation (list of lists). model representation of the data
     """
    file_name = train_file
    
    transcripts = Transcripts(train_file, subset=subset)
    
    if preprocess:
        preprocessed_name = train_file[:-5] + "_preprocessed.json"
        transcripts = rm_contractions_stopwords_and_lemmatize(transcripts, preprocessed_name)
        update_df(file_name, preprocessed_name)
        file_name = preprocessed_name
        
    
    model_representation = create_document_representation(transcripts=transcripts, representation=representation)

    make_json_model_and_data(file_name=file_name, transcripts=transcripts, 
    model = model_representation, representation = representation, comcast_data_type = comcast_data_type)

    return(model_representation)


class Transcripts:
    def __init__(self, train_file, subset):
        """
        Args:
          - train_file (str) path to train_file
          - subset (bool) if true, keep only first 100 and last 100
            words of transcript
        """
        self.train_file = train_file
        self.subset = subset

    def __iter__(self):
        if self.subset == True:
            for line in open(self.train_file):
                temp_list = []
                beginning = json.loads(line)['Transcript'].split(' ')[:100]
                end = json.loads(line)['Transcript'].split(' ')[-100:]
                combined = beginning + end
                combined_as_string = " ".join(combined)
                yield combined_as_string
        else:
            for line in open(self.train_file):
                yield json.loads(line)['Transcript']


# Did not use these, unless we want the preprocessed file returned in a different way

# def open_newline_delimited_json(filename):
#     """
#     Function Taken From: https://github.com/Cognosante/automated_processes/blob/master/helpers/gcp_tools.py
#     Args: filename (str)
#     Returns: list of dicts
#     """
#     records = []
#     with open(filename, "r") as infile:
#         for line in infile:
#             records.append(json.loads(line))
#     return records

def open_as_pandas(data):
    """
    Opens the data as a pandas DataFrame
    Args:
      - data (str). JSON file containing data
    Returns:
      - df (Pandas Dataframe). Pandas Dataframe containing the data
    """
    call_records = []
    with open(data, "r") as infile:
        for line in infile:
            call_records.append(json.loads(line))
    df = pd.DataFrame(call_records)
    return(df)

def update_df(data, new_transcripts):
    """
    Creates a new dataframe combining the necessary data with the preprocessed transcripts
    Args:
      - data (str). JSON file path name containing data
      - new_transcripts (str). JSON file path name containing preprocessed transcripts
    Returns:
    None
    """
    print(data)
    print(new_transcripts)
    old_df = open_as_pandas(data)
    transcripts = open_as_pandas(new_transcripts)
    
    old_df = old_df.drop(['Transcript'], axis=1)
    new_df = pd.concat([old_df, transcripts], axis = 1)
    
    combined_records = new_df.to_dict(orient='records')
    
    save_newline_delimited_json(combined_records, new_transcripts)
    
    


def save_newline_delimited_json(records, filename):
    """
    Function Taken From: https://github.com/Cognosante/automated_processes/blob/master/helpers/gcp_tools.py
    Saves list of Json-like Python objects into newline delimited json file
    Args:
      - records (list), json-like Python objects
      - filename (str). destination file
      - use_string (bool). tells whether or not to save data in a string format
    Returns:
    None
    """
    with open(filename, "w") as outfile:
        for record in records:
            outfile.write(json.dumps(record))
            outfile.write("\n")




def rm_contractions_stopwords_and_lemmatize(transcripts, new_name):
    """
    Removes contractions and stop words as well as lemmatizes the transcripts
    Args:
      - transcripts (obj). An object of the Class Transcripts
      - new_name (str). JSON file name that will be created to contain the updated data
    Returns:
      - Transcripts(new_name, False) (obj). An object of the class transcripts
    """
    contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have", 
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "i'd": "i would", "i'd've": "i would have","i'll": "i will",
                     "i'll've": "i will have","i'm": "i am","i've": "i have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}
    
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    def expand_contractions(text,contractions_dict=contractions_dict):
        """
        Expand Contractions portion of the main function is taken from the following:
        https://www.analyticsvidhya.com/blog/2020/04/beginners-guide-exploratory-data-analysis-text-data/
        Replaces the contraction with it's expanded version
        Args:
          - text (str). Transcript from a call
          - contractions_dict (dict). Dictionary of contractions as keys and their
              matching expanded version as the contraction
        Returns:
          - contractions_re.sub(replace, text) (str). Transcript with replaced contractions
        """
        def replace(match):
            """
            Finds matching expanded format for the given contraction
            Args: 
              - match (str). key from contractions_dict containing the contraction
                  that will need to be replaced
            Returns:
              - contractions_dict[match.group(0)] (str). value from contractions_dict 
                  containing expanded contraction
            """
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, text)

    nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])
    
    preprocessed_file = open(new_name, "w")
    
    for t in transcripts:
        temp_dict = {}
        t = t.replace('`','\'')
        t = expand_contractions(t)
        t = ' '.join([token.lemma_ for token in list(nlp(t)) if (token.is_stop==False)])
        temp_dict["Transcript"] = t
        temp_dict = json.dumps(temp_dict)
        preprocessed_file.write(temp_dict + '\n')
    
    return(Transcripts(new_name, False))


def create_document_representation(transcripts, representation):
    """
    Args:
      - transcripts (list or other iterable) iterable of transcripts, 
      - representation (str: {'bow', 'tfidf', 'lda', 'doc2vec'})    
    Returns:
      - Dependent on model chosen (list). model representation of data
    """

    def BagOfWords(transcripts):
        """
        Work Sited:
        https://www.tutorialspoint.com/gensim/gensim_creating_a_bag_of_words_corpus.htm
        Args: 
          - transcripts (list or other iterable) iterable of transcripts
        Returns:
          - BoW_words (list of lists of (word number, word frequency)). Bag of Words model 
              for all of the transcripts displaying the number of times each word appears in a transcript
          - dictionary (gensim Dictionary). Dictionary of words used in the data
        """
        transcript_tokenized = [simple_preprocess(t) for t in transcripts]
        dictionary =  Dictionary(transcript_tokenized)
        BoW_corpus = [dictionary.doc2bow(t, allow_update=True) for t in transcript_tokenized]
        id_words = [[(dictionary[id], count) for id, count in line] for line in BoW_corpus]
        model_name = "BOW_model_" + datetime.now().strftime("%d-%b-%Y_%H_%M_%S")
        dictionary.save(model_name)
        return(BoW_corpus, id_words, dictionary)

    
    def TF_IDF(bow_model, dictionary):
        """
        Work Sited:
        https://www.tutorialspoint.com/gensim/gensim_creating_tf_idf_matrix.htm
        Args:
          - bow_model (list of lists of (word number, word frequency)). Bag of Words model 
              for all of the transcripts displaying the number of times each word appears in a transcript
          - dictionary (gensim Dictionary). Dictionary of words used in the data
        Returns:
          - tfidf_words (list of lists). This is a list of lists containing the word and 
              it's TF-IDF score (ex: ['advise', 0.23])
        """
        tfidf = gensim.models.TfidfModel(bow_model)
        tfidf_words = []
        for doc in tfidf[bow_model]:
            doc_words = []
            for id, freq in doc:
                doc_words.append([dictionary[id], np.around(freq, decimals=2)])
            tfidf_words.append(doc_words)
        model_name = "tfidf_model_" + datetime.now().strftime("%d-%b-%Y_%H_%M_%S")
        tfidf.save(model_name)
        return(tfidf_words)

# There may be a problem because everything coming out of here has the same liklihood for each topic
    def Latent_Dirichlet_Allocation(transcripts, bow_model):
        """
        Work Sited:
        https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28
        Args:
          - transcripts (list or other iterable). iterable of transcripts
          - bow_model (list of lists). Bag of Words model for all of the transcripts displaying 
              the number of times each word appears in a transcript
        Returns:
          - train_vecs (list of lists). A list containing a list of the probability each transcript
              belongs in the indexed topic
        """
        num_top = 100
        lda = gensim.models.ldamulticore.LdaMulticore(corpus = bow_model, num_topics = num_top)
        train_vecs = []
        for i, t in enumerate(transcripts):
            top_topics = lda.get_document_topics(bow_model[i], minimum_probability=0.0)
            topic_vec = [top_topics[topic_idx][1] for topic_idx in range(num_top)]
            train_vecs.append(topic_vec)
        model_name = "LDA_model_" + datetime.now().strftime("%d-%b-%Y_%H_%M_%S")
        lda.save(model_name)
        float_list = []

        for vector in train_vecs:
            new_vector = [float(n) for n in vector]
            float_list.append(new_vector)
        return(float_list)

    def Document2Vector(transcripts):
        """
        Args:
          - transcripts (list or other iterable). iterable of transcripts
        Returns:
          - vectors (list of lists). List of each transcript's Doc2Vec vector location 
        """
        documents = [TaggedDocument(t, [i]) for i, t in enumerate(transcripts)]
        model = gensim.models.doc2vec.Doc2Vec()
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        vectors = []
        for t in transcripts:
            temp_array = model.infer_vector(simple_preprocess(t))
            x_str = np.array_repr(temp_array).replace('\n      ', '')
            vectors.append(x_str)
        model_name = "doc2vec_model_" + datetime.now().strftime("%d-%b-%Y_%H_%M_%S")
        model.save(model_name)
        return(vectors)


    if representation == 'bow':
        idx_words, words_num, dictionary = BagOfWords(transcripts)
        return(words_num)

    elif representation == 'tfidf':
        idx_words, words_num, dictionary = BagOfWords(transcripts)
        return(TF_IDF(idx_words, dictionary))

    elif representation == 'lda':
        idx_words,words_num, dictionary = BagOfWords(transcripts)
        return(Latent_Dirichlet_Allocation(transcripts, idx_words))

    elif representation == 'doc2vec':
        return(Document2Vector(transcripts))

# Would some sort of try except block be better here? or is there something similar?
    else:
        print("User Did Not Enter a Valid Representation")
        return(None)

def make_json_model_and_data(file_name, transcripts, model, representation, comcast_data_type):
    """
    Creates a json file containing the important/requested columns of the dataframe
    as well as the new model representation for each call
    Args:
      - file_name (str). file location of transcripts
      - transcripts (list or other iterable). iterable of transcripts
      - model (list of lists). model representation of the data
      - representation (str: {'bow', 'tfidf', 'lda', 'doc2vec'})
      - comcast_data_type (str: {'res', 'bus'}). string stating whether Comcast Business or Residential
          data is being used
    Returns:
    None
    """
    call_records = []
    with open(file_name, "r") as infile:
        for line in infile:
            call_records.append(json.loads(line))
    df = pd.DataFrame(call_records)
    
    # Creates a list of each main category grouping
    main_category_sections = {}
    subcategory_names = {}
    
    def extract_cats(cat_list):
        """
        Args:
          - cat_list (list) list of column title names
        Returns:
        None
        """
        for cat in cat_list:
            if cat not in subcategory_names:
                subcategory_names[cat] = None
            section_name = cat.split(".", 1)[0]
            
            if section_name not in main_category_sections:
                main_category_sections[section_name] = None
                
    #combines subcategories into their main category and marks entries as 1s and 0s
    def combine_subcategories(data, group):
        """
        Args:
          - data (Pandas DataFrame) Dataframe containg the necessary data
          - group (str). One Category from the main_category_sections list 
        """
        data[group] = data[[cat for cat in data.columns if cat.startswith(group)]].apply(lambda line: sum(line) > 0, axis = 1)
        data[group] = data[group].astype('int32')
    
    df = df.dropna(subset = ['categories'])
    
    _ = df['categories'].apply(extract_cats)
    main_category_sections = list(main_category_sections.keys())
    
    mlb = MultiLabelBinarizer()
    df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('categories')),\
                                           columns=mlb.classes_,index=df.index))
    for cat in main_category_sections:
        combine_subcategories(df, cat)
    
    if comcast_data_type == "res":
        wanted = ['agitation', 'average_confidence', 'percent_silence', 
                  'longest_silence', 'call_duration', 'silence_time', 
                  'call_type_detail', 'ne_call_duration', 'duration', 'word_count', 
                  'Transcript', 'Consent Types']
    if comcast_data_type == "bus":
        wanted = ['agitation', 'average_confidence', 'percent_silence', 
                  'silence_avg_duration', 'longest_silence', 
                  'silence_total_duration', 'silence_time', 'call_type_detail', 
                  'duration', 'word_count', 'Transcript', '06 Close']
    
    for cat in df.columns:
        if cat not in wanted:
            df = df.drop(columns = [cat])
            
    temp=pd.Series(model)
    model_data = pd.DataFrame(temp)
    
    combined_df = pd.concat([df, model_data], axis = 1)
    
    combined_df.rename(columns = {0: representation}, inplace = True)
    combined_records = combined_df.to_dict(orient='records')
    
    name = representation + '_' + file_name[:-5] + '_combined_data.json'
    name = name.replace('data\\', '')

    save_newline_delimited_json(combined_records, name)

    

def parse_command_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_file", type=str, 
    help = 'JSON file containing transcripts', 
    required = True, default=None)

    parser.add_argument("--subset", help='True if you only want to use the first 100 words and last 100 words', 
    required = False, default = False, action = "store_true")

    parser.add_argument("--preprocess", 
    help='Do you want the Data to be preprocessed? (i.e. lemmatization, contraction/stopword removal', 
    required = False, default = False, action = "store_true")

    parser.add_argument("--representation", type=str, 
    help='Pick which model you would like the data to be run through {"bow", "tfidf", "lda", "doc2vec"}', 
    required = True, default = None)

    parser.add_argument("--comcast_data_type", type=str, 
    help = "Is the data from Comcast Business or Residential? {'bus', 'res'}",
    required = True, default = None)

    args = parser.parse_args()

    return(args.train_file, args.subset, args.preprocess, args.representation, args.comcast_data_type)


if __name__ == "__main__":
    train_file, subset, preprocess, representation, comcast_data_type = parse_command_args()
    requested_data = main(train_file, subset, preprocess, representation, comcast_data_type)



# def bigramize(sentences):
#   #TODO, uncomment and use if using gensim implementation

#     """
#     Creates phrase model

#     """   
#     phrases = Phrases(sentences,min_count = 30)
#     phrases.save("phrases_model_072020")
#     bigram = Phraser(phrases)
#     bigram.save("bigram_model_072020")
#     return bigram