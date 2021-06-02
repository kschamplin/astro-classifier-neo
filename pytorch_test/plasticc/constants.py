# Constants for handling the plasticc dataset

label_map = {
    90: "SNIa",
    67: "SNIa-91bg",
    52: "SNIax",
    42: "SNII",
    62: "SNIbc",
    95: "SLSN-I",
    15: "TDE",
    64: "KN",
    88: "AGN",
    # 92: "RRL",
    # 65: "M-dwarf",
    # 16: "EB",
    # 53: "Mira",
    6: "μLens-Single",
    991: "μLens-Binary",
    992: "ILOT",
    993: "CaRT",
    994: "PISN",
    # 995: "μLens-String"
}
class_weights = {
    6: 170.16862186163797,
    15: 16.35778047109659,
    42: 0.22169645981674177,
    52: 3.4828115463325315,
    62: 1.266346729674999,
    64: 1667.140708915145,
    67: 5.516625140838312,
    88: 2.1861661370653325,
    90: 0.13358571703126057,
    95: 6.196683088863515,
    991: 416.0032162958992,
    992: 130.27597784119524,
    993: 22.90596221959858,
    994: 189.18917601170162
}
passband_map = {
    0: 'u',
    1: 'g',
    2: 'r',
    3: 'i',
    4: 'z',
    5: 'y'
}
# we define a dict of class_id: target mappings using a cheap hack.

label_targets = list(label_map.keys())
class_id_to_target = dict(zip(label_targets, range(len(label_targets))))

class_weights_target = {class_id_to_target[class_id]: weight for class_id, weight in class_weights.items()}
class_weights_target_list = torch.tensor([class_weights_target[x] for x in range(len(class_weights_target))])
