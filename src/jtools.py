### Get abbreviations for pmid dictionary
from nltk.stem import WordNetLemmatizer
import nltk
import re

from collections import Counter
from numpy import median

import pickle

#@Language.component("remove_trf_data")
#def remove_trf_data(doc):
    #doc._.trf_data = None
    #return doc
#nlp.add_pipe("remove_trf_data")
import gc
import torch
# Precompile the regex for efficiency
#tag_re = re.compile('<[^>]*>')
#non_word_re = re.compile('[\W]+')
#tag_re = re.compile('<[^>]*>')
#non_word_re = re.compile('[\W]+')

#def preprocessor(text):
    #if isinstance(text, str):
        #text = tag_re.sub(' ', text)
        #return non_word_re.sub(' ', text.lower()).strip()
    #elif isinstance(text, list):
        #return [non_word_re.sub(' ', tag_re.sub(' ', t).lower()).strip() for t in text]
    #return None
def preprocessor(text):
    if isinstance(text, str) and len(''.join([i.replace(" ", "") for i in text if not i.isdigit()])) >= 2:
        text = re.sub('<[^>]*>', ' ', text)
        # Exclude Greek characters from being replaced
        text = re.sub('[^\w\u0370-\u03FF\u1F00-\u1FFF]+', ' ', text.lower()).lstrip().rstrip()
        return text
    else:
        return None
def pmid2abr(texts):
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_sci_md", disable=["tagger", "parser"])
    nlp.add_pipe("abbreviation_detector")
    pmid_abrs = {}

    # Process texts in smaller batches to conserve memory
    for doc in nlp.pipe(texts.values(), batch_size=1000):
        pmid = next(key for key, value in texts.items() if value == doc.text)  # Find the corresponding PMID
        abrs = {str(abr): str(abr._.long_form) for abr in doc._.abbreviations}
        abrs = {abr: full for abr, full in abrs.items() if len(abr) < len(full)}
        pmid_abrs[pmid] = abrs

        # Clear memory more frequently if needed
        del doc
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()

    return pmid_abrs
def pmid2abr(texts):
    import spacy
    from scispacy.abbreviation import AbbreviationDetector
    from spacy.lang.en import English
    #,disable=["parser"])#"en_core_sci_scibert")"en_core_sci_lg"
    from spacy.language import Language
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_sci_lg", disable=["tagger", "parser"])
    nlp.add_pipe("abbreviation_detector")
    pmid_abrs = {}
    ##def nlp_pipe(texts_):
        ##try:
            ##doc = nlp.pipe(texts_)
        ##except:
            ##doc = ''
        ##return doc
    docs = nlp.pipe(list(texts.values()))
    ##docs = [nlp.pipe(i) for i in list(texts.values())]
    ##print(docs)
    ##del docs_
    ## texts is a list of texts
    texts = dict(zip(list(texts.keys()),docs))
    del docs
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()


    for pmid, doc in texts.items():


            ##doc = nlp(text)
            abrs = {str(abr): str(abr._.long_form) for abr in doc._.abbreviations}
            abrs = {abr:full for abr,full in abrs.items() if len(abr) < len(full)}
            pmid_abrs[pmid] = abrs
        ##except:
            ##pmid_abrs[pmid] = {'':''}

    return pmid_abrs
#, disable=["tagger", "parser"])
def pmid2noun(texts):
    import re
    import spacy
    from scispacy.abbreviation import AbbreviationDetector
    from spacy.lang.en import English
    #,disable=["parser"])#"en_core_sci_scibert")"en_core_sci_lg"
    from spacy.language import Language
    spacy.prefer_gpu()
    pmid_nouns = {}
    #def nlp_pipe(texts_):
        #try:
            #doc = nlp.pipe(texts_)
        #except:
            #doc = ''
        #return doc
    nlp = spacy.load("en_core_sci_lg")
    docs = nlp.pipe(list(texts.values()))
    #docs = [nlp.pipe(i) for i in list(texts.values())]
    #print(docs)
    #del docs_
    # texts is a list of texts
    texts = dict(zip(list(texts.keys()),docs))
    del docs
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()


    for pmid, doc in texts.items():
        tokenized = []
        try:
            #doc = nlp(text)
            abrs = list(set([preprocessor(str(i)) for i in doc.noun_chunks]))
            pmid_nouns[pmid] = abrs
        except:
            pmid_nouns[pmid] = []

    return pmid_nouns
def pmid2lemmas(texts):
    import spacy
    from scispacy.abbreviation import AbbreviationDetector
    from spacy.lang.en import English
    #,disable=["parser"])#"en_core_sci_scibert")"en_core_sci_lg"
    from spacy.language import Language
    spacy.prefer_gpu()
    pmid_lemmas = {}

    # Initialize Spacy model
    nlp = spacy.load("en_core_sci_lg")

    # Process the texts in a batch
    docs = nlp.pipe(list(texts.values()))

    # Create a dictionary mapping PMIDs to their corresponding processed docs
    texts = dict(zip(list(texts.keys()), docs))

    # Free up memory
    del docs
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()

    # Iterate through each document to get the lemmas
    for pmid, doc in texts.items():
        lemmatized = []
        try:
            # Iterate through each token in the document
            for token in doc:
                lemmatized.append(token.lemma_)
            pmid_lemmas[pmid] = lemmatized
        except:
            pmid_lemmas[pmid] = []

    return pmid_lemmas

# Example usage


#import spacy
import torch
import gc

def spacy_lemma(texts):
    # Initialize Spacy model
    import spacy
    from scispacy.abbreviation import AbbreviationDetector
    from spacy.lang.en import English
    #,disable=["parser"])#"en_core_sci_scibert")"en_core_sci_lg"
    from spacy.language import Language
    import spacy
    import torch
    import gc
    torch.set_default_dtype(torch.float16)
    spacy.prefer_gpu()

    nlp = spacy.load("en_core_sci_lg")

    # Process the texts
    processed_docs = list(nlp.pipe(texts))

    # Free up memory
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()

    # Initialize an empty list to store the lemmatized strings
    lemmatized_texts = []

    # Iterate over processed docs
    for doc in processed_docs:
        text = doc.text  # original text

        # Pick the last token, but check if it is a punctuation
        last_token = doc[-1] if len(doc[-1].text) > 1 or not doc[-1].is_punct else doc[-2]

        # Replace only the last (eligible) token in the original text with its lemma, using character indices
        text = text[:last_token.idx] + last_token.lemma_ + text[last_token.idx + len(last_token.text):]

        # Append the lemmatized string to the list
        lemmatized_texts.append(text)

    return lemmatized_texts

pronouns,all_stops,cancer_stops = pickle.load(open('pronouns,all_stops,cancer_stops.pickle','rb'))
def strip_numbers_and_spaces(s):
    s = s.replace(" ", "").lower()
    return ''.join([char for char in s if not char.isdigit()])
def soft_clean(bcarc,pronouns=pronouns,all_stops=all_stops,cancer_stops=cancer_stops):
    # Convert lists to sets for faster membership tests
    pronouns_set = set(pronouns)
    all_stops_set = set(all_stops)
    cancer_stops = set(cancer_stops)


    if not (type(bcarc) is str and bcarc):
        return None

    # Simplify the string operation
    cleaned_bcarc = ''.join([i for i in bcarc if not i.isdigit() and i != ' '])
    if len(cleaned_bcarc) < 2:
        return None

    if bcarc.startswith(tuple(pronouns)):
        bcarc = " ".join(bcarc.split(" ")[1:])
    bcarc_s = strip_numbers_and_spaces(bcarc)
    if bcarc not in pronouns_set and bcarc not in all_stops_set and not any(c in bcarc for c in cancer_stops) and bcarc_s not in pronouns_set and bcarc_s not in all_stops_set and not any(c in bcarc_s for c in cancer_stops):
        #pre_bcarc = str(preprocessor(bcarc))
        return bcarc#pre_bcarc

    else:
        return None

def pluralize(token):

    import inflect
    p = inflect.engine()
    return p.plural(token)


def flat2gen(alist):
  for item in alist:
    if isinstance(item, list):
      for subitem in item: yield subitem
    else:
      yield item
def flatten_once(xss):
    "Converts the list of lists into one list"
    return [x for xs in xss for x in xs]
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def nuance_lemmatizer(token):
    if " " in token:
        token = token.split(" ")
        token[-1] = lemmatizer.lemmatize(token[-1])
        token = " ".join(token)
    else:
        token = lemmatizer.lemmatize(token)
    return token


import unicodedata
import re
def remove_quotes(text):
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    elif text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    else:
        return text
def normalize_characters(text):
    # Normalize Greek characters
    greek_chars = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'ς', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω', 'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω']
    for char in greek_chars:
        text = text.replace(char, unicodedata.normalize('NFC', char))

    # Normalize space characters
    space_chars = ['\xa0', '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200a', '\u202f', '\u205f', '\u3000']
    for space in space_chars:
        text = text.replace(space, ' ')

    # Normalize single quotes
    single_quotes = ['‘', '’', '‛', '′', '‹', '›', '‚', '‟']
    for quote in single_quotes:
        text = text.replace(quote, "'")

    # Normalize double quotes
    double_quotes = ['“', '”', '„', '‟', '«', '»', '〝', '〞', '〟', '＂']
    for quote in double_quotes:
        text = text.replace(quote, '"')

    # Normalize brackets
    brackets = {
        '【': '[', '】': ']',
        '（': '(', '）': ')',
        '｛': '{', '｝': '}',
        '〚': '[', '〛': ']',
        '〈': '<', '〉': '>',
        '《': '<', '》': '>',
        '「': '[', '」': ']',
        '『': '[', '『': ']',
        '〔': '[', '〕': ']',
        '〖': '[', '〗': ']'
    }
    for old, new in brackets.items():
        text = text.replace(old, new)

    # Normalize hyphens and dashes
    hyphens_and_dashes = ['‐', '‑', '‒', '–', '—', '―']
    for dash in hyphens_and_dashes:
        text = text.replace(dash, '-')

    # Normalize line breaks
    line_breaks = ['\r\n', '\r']
    for line_break in line_breaks:
        text = text.replace(line_break, '\n')

    # Normalize superscripts and subscripts to normal numbers
    superscripts = '⁰¹²³⁴⁵⁶⁷⁸⁹'
    subscripts = '₀₁₂₃₄₅₆₇₈₉'
    normal_numbers = '0123456789'

    for super_, sub_, normal in zip(superscripts, subscripts, normal_numbers):
        text = text.replace(super_, normal).replace(sub_, normal)

    # Remove or normalize any remaining special characters using the 'NFKD' method
    text = unicodedata.normalize('NFKD', text)

    return remove_quotes(text)

def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]
#def preprocessor(text):
    #if isinstance((text), (str)) and len(''.join([i.replace(" ", "") for i in text if not i.isdigit()])) >= 2:
        #text = re.sub('<[^>]*>', ' ', text)
        #text = re.sub('[\W]+', ' ', text.lower()).lstrip().rstrip()
        #return text
    #else:
        #pass
    
    
  
def tokenbinder(new_ghit,e2,e2_out,g_counts,mean_counts,mean_score,meta_data_text,e2_tokens):
    import xlsxwriter
    import os
    
    from datetime import date

    today = date.today()

    dout = today.strftime("%m-%d-%y")
    path = "/home/joneill/PCarD_data_"+str(dout)+"/"
    isExist = os.path.exists(path)
    if not isExist:

        os.makedirs(path)

    e2_out.sort(key = lambda x: x[3],reverse=True)
    bert_book = xlsxwriter.Workbook(path+ 'PCaRD_'+meta_data_text +'.xls')
    bold = bert_book.add_format({'bold': True})
    sheet2 = bert_book.add_worksheet("CarD-BERT")

    headers = ['G1','G2A','G2B','G3']
    columns = ['Found(p)','Publications/Group','Publications(x̄[±σ])','p-val']

    
    sheet2.write(0,0,meta_data_text +" PCarD - Unique Tokens=" + str(len(e2_tokens)) ,bold)
    sheet2.write(1,0,"IARC",bold)
    col = 1
    for q,h in enumerate(headers):
        m = q+1
        sheet2.write(col,m,h)
        
    for q,h in enumerate(columns):
        m = q+2
        sheet2.write(m,0,h)
    col = col +1
    for d,prop in enumerate(e2[0]):
        

        sheet2.write(col,d+1,str(round(prop,3)))
    #sheet2.write(col,d+2,str(len(e2_tokens)))

    col = col +1
    for d,n in enumerate(g_counts):
        

        sheet2.write(col,d+1,str(n))    

    col = col +1
    for d,n in enumerate(mean_counts):
        

        sheet2.write(col,d+1,str(n))   

    col = col +1
    for d,s in enumerate(mean_score):
        

        sheet2.write(col,d+1,str(s))
    #if g1y is not False:
        #sheet2.write(1,7,"Native Accuracy")
        #sheet2.write(2,7,"G1*0."+trained)
        #sheet2.write(2,8,"G2A*0."+trained)
        #sheet2.write(3,7,str(native_accuracy[0]))
        #sheet2.write(3,8,str(native_accuracy[1]))

                 


    #col = col +1
    col = 0
    for e,groups_ in enumerate(new_ghit):
        sheet3 = bert_book.add_worksheet(headers[e])
        #sheet3.col(0).width = 7000
        sheet3.write(0,0,'IARC '+ headers[e]+" Tokens(n)="+str(g_counts[e]),bold)#,header_style)
        sheet3.write(0,1,'Found(True/False)',bold)#,header_style)
        sheet3.write(0,2,'p-val ',bold)
        sheet3.write(0,3,'Publications(n)',bold)#,header_style)

        #sheet3.write(0,5,'Years[n] ' + headers[e],bold)
        #sheet3.write(0,6,'PMID(s) ' + headers[e],bold)
        col = 0
        for f, gr in enumerate(groups_):
            col = col+1
            token_format = bert_book.add_format()
            token_format.set_bg_color(gr[-1])
            word_format = bert_book.add_format()
            word_format.set_bg_color(gr[-3])
            score_format = bert_book.add_format()
            score_format.set_bg_color(gr[-2])
            sheet3.write(col,0,gr[0],token_format)
            sheet3.write(col,1,gr[1])
            sheet3.write(col,2,str(gr[3]),score_format)
            sheet3.write(col,3,str(gr[2]),word_format)
            #sheet3.write(col,4,gr[6])
            #sheet3.write(col,5,gr[5])
            #sheet3.write(col,6,gr[4])


    sheet4 = bert_book.add_worksheet('Tokens Found')
    #sheet4.col(0).width = 7000
    sheet4.write(0,0,"Token(s) Found",bold) ### (row_n, column_n, string)  
    sheet4.write(0,1,'IARC')
    sheet4.write(0,2,'p-val')
    sheet4.write(0,3,'Publicatoins')
    #sheet4.write(0,4,'Year(s)')
    #sheet4.write(0,5,'PMID(s)')
    #12/9/22 update to contain sheet3 0,2-5
    col = 0
    for datums in e2_out:
        col = col+1
        token_format = bert_book.add_format()
        token_format.set_bg_color(datums[-1])
        word_format = bert_book.add_format()
        word_format.set_bg_color(datums[-3])
        score_format = bert_book.add_format()
        score_format.set_bg_color(datums[-2])        
        
        sheet4.write(col,0,datums[0],token_format)
        sheet4.write(col,1,datums[1])
        sheet4.write(col,2,datums[2],score_format)
        sheet4.write(col,3,datums[3],word_format)
        #sheet4.write(col,4,datums[4])
        #sheet4.write(col,5,datums[5])


    bert_book.close()
