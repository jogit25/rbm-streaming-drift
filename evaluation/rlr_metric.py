def compute_rlr(clean_accuracy, attack_accuracy):

    if clean_accuracy == 0:
        return 0

    return attack_accuracy / clean_accuracy