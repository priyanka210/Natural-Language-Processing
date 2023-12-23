import gzip, re, sys

def tokenize(l):
    
    # Define a regular expression pattern for tokens
    token_pattern = r"[\w$'\-]+|[.,!?;:\"()\[\]\{\}&@~\+<>%]+"
    
    # Tokenize the input text using the regular expression pattern
    tokens = re.findall(token_pattern, l)
    
    return tokens
    

# The first argument is the filename to work with
fname = sys.argv[1]

# Use gzip.open to open a compressed file
with gzip.open(fname, mode='rt', encoding='utf-8') as infile:
    
    for line in infile:
        # Tokenize each sentence/line
        tokens = tokenize(line)
        
        # Print each token on a separate line
        for t in tokens:
            print(t)
