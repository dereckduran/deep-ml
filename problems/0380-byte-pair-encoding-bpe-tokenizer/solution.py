def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """
    merges = []
    for i in range(num_merges):
        pair_counts = {}
        for word in corpus:
            letter_list = word.split()
            for i in range(len(letter_list) - 1):
                pair = (letter_list[i], letter_list[i+1])
                if pair in pair_counts:
                    pair_counts[(letter_list[i], letter_list[i+1])] += corpus[word]
                else:
                    pair_counts[(letter_list[i], letter_list[i+1])] = corpus[word] 
            
        best_pair = ' '.join(max(pair_counts, key=pair_counts.get))
        merged_pair = ''.join(max(pair_counts, key=pair_counts.get))
        merges.append(max(pair_counts, key=pair_counts.get))



        for word in list(corpus):
            new_word = word.replace(best_pair, merged_pair)
           
            corpus[new_word] = corpus.pop(word)
    return merges
            