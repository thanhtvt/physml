def split_dictionary(dictionary, ratio: float):
    # Calculate the number of values for each part based on the ratio
    total_values = len(dictionary) * len(dictionary[list(dictionary.keys())[0]])
    part1_values = int(total_values * ratio)

    # Create two new dictionaries for the parts
    part1 = {}
    part2 = {}
    num_keys = len(dictionary.keys())
    part1_portion = part1_values // num_keys

    # Iterate over the keys in the original dictionary
    for key in dictionary.keys():
        values = dictionary[key]

        # Split the values based on the calculated counts
        part1[key] = values[:part1_portion]
        part2[key] = values[part1_portion:]

    return part1, part2
