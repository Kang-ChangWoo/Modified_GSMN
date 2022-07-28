from vocab import Vocabulary
import evaluation
import os

#added
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#RUN_PATH = "models1/model_best.pth.tar"
RUN_PATH = "/root/dev/Modified_GSMN/pre-trained/model_dense_f30k.pth.tar"
DATA_PATH = "/root/dev/Modified_GSMN/data"
evaluation.evalrank(RUN_PATH, data_path=DATA_PATH, split="test")
