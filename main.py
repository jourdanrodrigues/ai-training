import sys
from os import path

from app.loader import Loader

DATA_PATH = path.join(path.dirname(path.abspath(__file__)), ".data")

args = set(sys.argv[1:])

if __name__ == "__main__":
    loader = Loader(data_path=DATA_PATH)
    if "--train" in args:
        loader.perform_train(loops=2)
    elif "--test" in args:
        loader.load_saved_model()
        loader.test_network()
