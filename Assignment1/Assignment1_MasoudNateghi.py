# Load libraries
import openpyxl
import Levenshtein
from fuzzywuzzy import fuzz
import string
import re
import nltk
import itertools
from nltk.tokenize import sent_tokenize
# %%
# Specify the path to your Excel file
excel_file_path = "Assignment1GoldStandardSet.xlsx"

# Load the Excel workbook
workbook = openpyxl.load_workbook(excel_file_path)

# Access a specific sheet by name
sheet_name = "Sheet1"
sheet = workbook[sheet_name]

# Read all the reddit posts
texts = []
for i in range(2, 36):
    texts.append(sheet['B'+str(i)].value.lower())
# %% 
# Read Sympthom Lexicon files
lexicon_dict = {}
symptom_dict = {}
infile = open('./COVID-Twitter-Symptom-Lexicon.txt')
for line in infile:
    items = line.split('\t')
    # Build lexicon dictionary
    if items[1] in lexicon_dict:
        lexicon_dict[items[1]].append(items[-1].strip().lower()) # strip for \n at the end of lexicon/ lowercase
    else: 
        lexicon_dict[items[1]] = [items[-1].strip().lower()] # strip for \n at the end of lexicon/ lowercase
    
    # Build Symptom dictionary
    if not items[1] in symptom_dict:
        symptom_dict[items[1]] = items[0]
# %%
def run_sliding_window_through_text(words, window_size):
    """
    Generate a window sliding through a sequence of words
    """
    word_iterator = iter(words) # creates an object which can be iterated one element at a time
    word_window = tuple(itertools.islice(word_iterator, window_size)) #islice() makes an iterator that returns selected elements from the the word_iterator
    yield word_window
    #now to move the window forward, one word at a time
    for w in word_iterator:
        word_window = word_window[1:] + (w,)
        yield word_window
# %%
def match_dict_similarity(text, CUI):
    match_list = None
    max_similarity_obtained = -1
    best_match = ''
    words = list(nltk.word_tokenize(text))
    sents = list(nltk.sent_tokenize(text))
    #go through each expression
    expressions = lexicon_dict[CUI]
    for exp in expressions:
        
        #create the window size equal to the number of word in the expression in the lexicon
        size_of_window = len(exp.split())
        # set the treshold
        thr = max(1 - 0.07 * len(exp.split()), 0.8)
        for window in run_sliding_window_through_text(words, size_of_window):
            window_string = ' '.join(window)

            
            similarity_score = Levenshtein.ratio(window_string, exp)
            
            if similarity_score >= thr:

                # print("{:<20} {:<20} {:<25} {:<8}".format(similarity_score, exp, window_string, CUI))
                if similarity_score>max_similarity_obtained:
                    max_similarity_obtained = similarity_score
                    best_match = window_string
                    best_sent = ''
                    pattern = re.escape(window_string)
                    for sent in sents:
                        for match in re.finditer(pattern, sent):
                            match_list = [sent, CUI, window_string, exp, match.start(), match.end()]
        
    return match_list 
# %%
text_symptom_dict = {}
i = 0
for text in texts:
    text_symptom_dict[i] = []
    print(i)
    for CUI in lexicon_dict.keys():
        match_list = match_dict_similarity(text, CUI)
        if not match_list == None:
            text_symptom_dict[i].append(match_list)
    i += 1
# %%
def in_scope(neg_end, text,symptom_expression):
    '''
    Function to check if a symptom occurs within the scope of a negation based on some
    pre-defined rules.
    :param neg_end: the end index of the negation expression
    :param text:
    :param symptom_expression:
    :return:
    '''
    negated = False
    text_following_negation = text[neg_end:]
    tokenized_text_following_negation = list(nltk.word_tokenize(text_following_negation))
    # this is the maximum scope of the negation, unless there is a '.' or another negation
    three_terms_following_negation = ' '.join(tokenized_text_following_negation[:min(len(tokenized_text_following_negation),4)])
    #Note: in the above we have to make sure that the text actually contains 3 words after the negation
    #that's why we are using the min function -- it will be the minimum or 3 or whatever number of terms are occurring after
    #the negation. Uncomment the print function to see these texts.
    #print (three_terms_following_negation)
    match_object = re.search(symptom_expression,three_terms_following_negation)
    if match_object:
        period_check = re.search('\.',three_terms_following_negation)
        next_negation = 1000 #starting with a very large number
        #searching for more negations that may be occurring
        for neg in negations:
            # a little simplified search..
            if re.search(neg,text_following_negation):
                index = text_following_negation.find(neg)
                if index<next_negation:
                    next_negation = index
        if period_check:
            #if the period occurs after the symptom expression
            if period_check.start() > match_object.start() and next_negation > match_object.start():
                negated = True
        else:
            negated = True
    return negated
# %%
#loading the negation expressions
negations = []
infile = open('./neg_trigs.txt')
for line in infile:
    negations.append(str.strip(line))
    
# %%
for i in range(len(text_symptom_dict)):
    for mt in text_symptom_dict[i]:
        is_negated = False
        #Note: I broke down the code into simpler chunks for easier understanding..
        text = mt[0]
        cui = mt[1]
        expression = mt[2]
        start = mt[3]
        end = mt[4]
        #uncomment the print calls to separate each text fragment..
        #print('=------=')

        #go through each negation expression
        for neg in negations:
            #check if the negation matches anything in the text of the tuple
            for match in re.finditer(r'\b'+neg+r'\b', text):
                #if there is a negation in the sentence, we need to check
                #if the symptom expression also falls under this expression
                #it's perhaps best to pass this functionality to a function.
                # See the in_scope() function
                is_negated = in_scope(match.end(),text,expression)
                if is_negated:
                    cui += 'neg'
                    mt[1] = cui
                    break
# %%
file_dict = {}
for i in range(len(text_symptom_dict)):
    symptom_exps = '$$$'
    std_symptom = '$$$'
    symptom_CUIs = '$$$'
    neg_flag = '$$$'
    for j in range(len(text_symptom_dict[i])):
        symptom_exps += (text_symptom_dict[i][j][2] + '$$$')
        std_symptom += (text_symptom_dict[i][j][3] + '$$$')
        symptom_CUIs += (text_symptom_dict[i][j][1][:8] + '$$$')
        if text_symptom_dict[i][j][1].endswith('neg'):
            neg_flag += '1$$$'
        else:
            neg_flag += '0$$$'
    file_dict[i] = [symptom_exps, std_symptom, symptom_CUIs, neg_flag]
# %%
file_path = 'result.xlsx'
# Load the existing Excel workbook
workbook = openpyxl.load_workbook(file_path)
# Choose a specific sheet to work with (if it exists)
sheet = workbook.active  # This selects the default active sheet
for i in range(len(file_dict)):
    sheet['C'+str(i+2)] = file_dict[i][2]
    sheet['D'+str(i+2)] = file_dict[i][3]
    
workbook.save(file_path)
























