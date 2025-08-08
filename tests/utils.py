import numpy as np

def compare_histograms(hist1, hist2):
    # compare two histograms
    for key in hist1.keys():
        if key not in hist2:
            return False, {
                "error": "Key not found in second histogram",
                "key": key,
                "value1": hist1[key],
                "value2": None
            }
        if not np.isclose(hist1[key], hist2[key]):
            return False, {
                "error": "Value differs in histograms",
                "key": key,
                "value1": hist1[key],
                "value2": hist2[key]
            }
    return True, {}

def compare_stats_exact_match(reference, result):
    
    for i in range(len(result)):
        h1 = result[i].get("histogram", {})
        h2 = reference[i].get("histogram", {})
        correct, errors = compare_histograms(h1, h2)
        if not correct:
            errors["index"] = i
            return correct, errors

        for key in result[i].keys():
            if key == "histogram":
                continue
            b2 = reference[i][key]
            value = result[i][key]
            
            if not np.allclose(b2, value):
                if key == "majority" or key == "minority":
                    # ignore tie cases
                    m1 = h1[value]
                    m2 = h2[value]
                    if m1 == m2:
                        continue
                    
                return False, {
                    "index": i,
                    "key": key,
                    "value": value,
                    "truth": b2,
                }
    return True, {}
