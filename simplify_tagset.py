# Based on the Brown corpus tagset.
# Set a named parameter to `True` if you wish to keep all distinct tag types for that category.
# param `tagged_words` is expected to be an iterable (list or otherwise) of word-tag tuples.
def simplify_tagset(tagged_words, \
                    keep_det_dst = False,   # Keep determiner distinctions \ 
                    keep_fw_dst = False,    # Keep foreign word distinctions \
                    keep_adj_dst = False,   # Keep adjective distinctions \
                    keep_modaux_dst = False,# Keep modal auxiliary distinctions \
                    keep_noun_dst = False,  # Keep noun distinctions \
                    keep_noun_pl_dst = False, # Keep noun plurality distinctions \
                    keep_pron_dst = False,  # Keep pronoun distinctions \
                    keep_adv_dst = False,   # Keep adverb distinctions \
                    keep_verb_dst = False,  # Keep verb distinctions \
                    keep_wh_dst = False,    # Keep WH-word distinctions \
                    ):

    rtn = []
    for word, tag in tagged_words:

        if not tag:
            rtn.append((word, "None"))
            continue

        if not keep_det_dst:
            if tag[0] == 'A':
                tag = 'A'
            elif tag[:2] == 'DT':
                tag = 'DT'
        if not keep_fw_dst:
            if tag[:2] == 'FW':
                tag = 'FW'
        if not keep_adj_dst:
            if tag[:2] == 'JJ':
                tag = 'JJ'
        if not keep_modaux_dst:
            if tag[:2] == 'MD':
                tag = 'MD'
        if not keep_noun_dst:
            t = tag
            if t[0] == 'N':
                tag = 'N'
                if keep_noun_pl_dst and 'S' in t:
                    tag += 'S'
        if not keep_pron_dst:
            if tag[0] == 'P':
                tag = 'P'
        if not keep_adv_dst:
            if tag[0] == 'R':
                tag = 'R'
        if not keep_verb_dst:
            if tag[0] in ['B', 'H', 'V'] or tag[:2] == 'DO':
                tag = 'V'
        if not keep_wh_dst:
            if tag[0] == 'W':
                tag = 'W'

        rtn.append((word, tag))

    return rtn