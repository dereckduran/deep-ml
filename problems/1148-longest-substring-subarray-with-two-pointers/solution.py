def longest_unique_substring(s):
    # s: a string
    # return an integer length
    if len(s) == 0:
        return 0
    
    # two pointer technique
    '''
    start two pointers at the beginning
    move one ahead
    if the letters within that substring arent unique
    move the left pointer forward
    keep track of the max distance between right - left + 1

    '''
    left, right = 0, 1
    max_distance = 0
    idx = {}
    for i in range(len(s)):
        '''if s[right] in idx:
            idx[s[left]] = i
            
        idx[s[i]] = left
        print'''
        if s[i] in idx:
            left = idx[s[i]] + 1
        idx[s[i]] = i
        max_distance = max(max_distance, right - left)
        right += 1

    return max_distance