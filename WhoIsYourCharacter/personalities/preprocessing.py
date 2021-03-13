def tokenize(line):
    line = line.lower()
    tokens = nltk.word_tokenize(re.sub(punctuation, ' ', line))
    return tokens, len(tokens)

def load_southpark(in_file):
    # Load input file
    df = pd.read_csv(in_file)
    df = df[['Character', 'Line']]
    df = df.rename(columns={ 'Character': 'speaker', 'Line': 'text' })

    df['source'] = 'South Park'
    df['gender'] = None
    df['age'] = None

    # Tokenize text column
    df['tokenized'], df['length'] = zip(*df['Line'].apply(tokenize))

    return df

def load_friends(in_file):
    with open(in_file) as f:
        content = f.readlines()
        
    # Load json lines and keep columns of interest
    df = pd.DataFrame([json.loads(line) for line in content])
    df = df[['speaker', 'text']]
    df = df[df['speaker'] != 'TRANSCRIPT_NOTE']

    # Tokenize text column
    df['tokenized'], df['length'] = zip(*df['text'].apply(tokenize))

    df['source'] = 'Friends'
    df['gender'] = None
    df['age'] = None

    return df

def load_movies(lines_file, meta_file):
    # Load dataframes
    kwargs = { 'delimiter': '\ \+\+\+\$\+\+\+\ ', 'encoding' : 'latin-1', 'header' : None }
    line_df = pd.read_csv(lines_file, **kwargs)
    meta_df = pd.read_csv(meta_file, **kwargs)

    # Set column names
    line_df.columns = ['line_id','char_id','movie_id','char_name','text']
    meta_df.columns = ['char_id','char_name','movie_id', 'movie_name','gender','credits_pos']

    # Fromalize meta dataframe
    meta_df['gender'] = meta_df['gender'].str.strip()
    meta_df = meta_df[meta_df['gender'] != '?']
    meta_df['gender'] = meta_df['gender'].apply(lambda x: 0 if x in ['m', 'M'] else 1) 

    # Merge meta and lines df's on movie_id and char_name
    df = pd.merge(line_df, meta_df, how='inner', on=['movie_id', 'char_name'],
                  left_index=False, right_index=False, sort=True, copy=False,
                  indicator=False).drop('char_id_y', axis=1)
    df = df.rename(columns={ 'char_id_x': 'char_id' })

    # Cleanup final df
    df['age'] = df['char_id'].apply(lambda x: MOVIE_CHAR_TO_AGE[x], axis=1)
    df = df[['movie_name', 'char_name', 'text', 'gender']]
    df = df.rename(columns={ 'movie_name': 'source', 'char_name': 'speaker' })

    # Tokenize text column
    df['tokenized'], df['length'] = zip(*df['text'].apply(tokenize))

    return df