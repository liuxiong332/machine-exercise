import numpy as np
import pandas as pd
import re
import gensim

def read_file():
  df = pd.read_csv('./HillaryEmails.csv')
  return df[['Id', 'ExtractedBodyText']].dropna()

def clean_email_texts(text):
  text = text.replace('\n', ' ')
  text = re.sub(r'-', ' ', text)
  text = re.sub(r'\d+/\d+/\d+', '', text)
  text = re.sub(r'[0-2]?[0-9]:[0-6][0-9]', '', text)
  text = re.sub(r'\w+@\w+\.\w+', '', text)
  text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) #网址，没意义
  pure_text = ''

  for letter in text:
    if letter.isalpha() or letter == ' ':
      pure_text += letter
  text = ' '.join(word for word in pure_text.split() if len(word) > 1)
  return text


stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours', 
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their', 
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once', 
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you', 
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will', 
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be', 
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself', 
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both', 
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn', 
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about', 
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn', 
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']
            

def process_texts(df):
  docs = df['ExtractedBodyText']
  return docs.apply(lambda x: clean_email_texts(x))

def train_texts(docs):
  texts = [[word for word in doc.lower().split() if word not in stoplist] for doc in docs]
  dictionary = gensim.corpora.Dictionary(texts)
  corpus = [dictionary.doc2bow(text) for text in texts]
 
  lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
  print(lda.print_topics(num_topics=20, num_words=5))
  return dictionary, lda

df = read_file()
docs = process_texts(df)
dictionary, lda = train_texts(docs)
new_texts = [
  'To all the little girls watching...never doubt that you are valuable and powerful & deserving of every chance & opportunity in the world.',
  'I was greeted by this heartwarming display on the corner of my street today. Thank you to all of you who did this. Happy Thanksgiving.',
  'Hoping everyone has a safe & Happy Thanksgiving today, & quality time with family & friends.',
  'Scripture tells us: Let us not grow weary in doing good, for in due season, we shall reap, if we do not lose heart.',
  'Let us have faith in each other. Let us not grow weary. Let us not lose heart. For there are more seasons to come and...more work to do',
  'We have still have not shattered that highest and hardest glass ceiling. But some day, someone will',
  'To Barack and Michelle Obama, our country owes you an enormous debt of gratitude. We thank you for your graceful, determined leadership',
  'Our constitutional democracy demands our participation, not just every four years, but all the time',
  'You represent the best of America, and being your candidate has been one of the greatest honors of my life',
  'Last night I congratulated Donald Trump and offered to work with him on behalf of our country',
  "Already voted? That's great! Now help Hillary win by signing up to make calls now',"
  "It's Election Day! Millions of Americans have cast their votes for Hillary—join them and confirm where you vote",
  "We don’t want to shrink the vision of this country. We want to keep expanding it",
  "We have a chance to elect a 45th president who will build on our progress, who will finish the job",
  "I love our country, and I believe in our people, and I will never, ever quit on you. No matter what"
]
new_texts = map(lambda x: clean_email_texts(x), new_texts)
new_texts = [[word for word in doc.lower().split() if word not in stoplist] for doc in new_texts]
corpus = [dictionary.doc2bow(text) for text in new_texts]
for text in corpus:
  print(lda.get_document_topics(text))