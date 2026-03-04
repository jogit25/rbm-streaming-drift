import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from datasets.sea_stream import SEAStream
from datasets.sudden_stream import SuddenStream
from datasets.recurrent_stream import RecurrentStream

from attacks.concept_poisoning import ConceptPoisoning

from models.robust_rbm import RobustSupervisedRBM
from drift.drift_detector import RRBMDriftDetector
from evaluation.prequential import PrequentialEvaluator

from evaluation.result_logger import ResultLogger
from evaluation.rlr_metric import compute_rlr


STREAM_LENGTH = 10000
logger = ResultLogger()

streams = {
    "gradual": SEAStream,
    "sudden": SuddenStream,
    "recurrent": RecurrentStream
}

concept_counts = [1,2,3,4,5]


for drift_name, stream_class in streams.items():

    print("\n==========================")
    print("Experiment 2 — drift:", drift_name)
    print("==========================")

    # CLEAN BASELINE
    stream = stream_class()

    rbm = RobustSupervisedRBM(3,5,2)
    detector = RRBMDriftDetector()
    evaluator = PrequentialEvaluator(detector)

    for i in range(STREAM_LENGTH):

        if i % 1000 == 0:
            print("Processed:", i)

        x,y = stream.next_instance()

        x_dict = {f"f{i}": float(x[i]) for i in range(3)}
        v_tensor = torch.from_numpy(x).float().unsqueeze(0)

        evaluator.process_instance(x_dict,y,v_tensor,rbm)

    clean_accuracy = evaluator.get_accuracy()

    print("Clean accuracy:", clean_accuracy)


    # CONCEPT ATTACK
    for n in concept_counts:

        print("\nConcept attack:", n)

        stream = stream_class()
        attack = ConceptPoisoning(n, STREAM_LENGTH)

        rbm = RobustSupervisedRBM(3,5,2)
        detector = RRBMDriftDetector()
        evaluator = PrequentialEvaluator(detector)

        for i in range(STREAM_LENGTH):

            if i % 1000 == 0:
                print("Processed:", i)

            x,y = stream.next_instance()
            x,y = attack.poison(x,y)

            x_dict = {f"f{i}": float(x[i]) for i in range(3)}
            v_tensor = torch.from_numpy(x).float().unsqueeze(0)

            evaluator.process_instance(x_dict,y,v_tensor,rbm)

        accuracy = evaluator.get_accuracy()

        rlr = compute_rlr(clean_accuracy, accuracy)

        print("Accuracy:", accuracy)
        print("RLR:", rlr)

        logger.log(
            dataset="SEA",
            drift_type=drift_name,
            attack_type="concept",
            attack_ratio=n,
            accuracy=accuracy,
            rlr=rlr
        )