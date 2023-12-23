import sys

# The first argument is the filename to work with
fname = sys.argv[1]

# Open a token file in read only mode
with open(fname, mode='r', encoding='utf-8') as infile:

    # Define a dictionary to store tokens and their frequency
    token_counter = {} 

    # Iterate over the file
    for line in infile:
        
        # Tokenize each line and convert to lowercase
        token = line.strip().lower()

        # Check if the token is present in the dict
        # if yes - increment the count by 1
        # if no - assign count as 1    
        if token in token_counter:
            token_counter[token] += 1
        else:
            token_counter[token] = 1

    #print(len(token_counter))
    #print(sum(token_counter.values()))
    #print(sum(1 for count in token_counter.values() if count == 1))

    # Sort the tokens by frequency in non-ascending order
    sorted_tokens = sorted(token_counter.items(), key=lambda item: item[1], reverse=True)[:20]
        
    for token, frequency in sorted_tokens:
        print(f"{token} {frequency}")
