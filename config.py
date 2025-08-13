from pathlib import Path

DATA_PATH = Path('data') /'pad-ufes-20'
PAD_20_PATH = Path("/home/pedrobouzon/life/datasets/pad-ufes-20")
PAD_20_IMAGES_FOLDER = PAD_20_PATH / "images"
PAD_20_ONE_HOT_ENCODED = DATA_PATH/ "pad-ufes-20-one-hot-demo.csv"
PAD_20_RAW_METADATA = PAD_20_PATH / "metadata.csv"
PAD_20_SENTENCE_ENCODED = DATA_PATH / "pad-ufes-20-sentence-demo.csv"
PAD_20_BAYESIAN_DATA = DATA_PATH


# vision baselines results path (for bayesian network)
IMAGE_ONLY_EXPERIMENT = 174692956751647 # change this to match the folder of the results of your image-only model
