import logging
from pathlib import Path
from shutil import rmtree


BASE_PATH = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = Path.joinpath(BASE_PATH, "model.h5")
LOGS_PATH = Path.joinpath(BASE_PATH, ".logs")
ASSETS_PATH = Path.joinpath(BASE_PATH, "assets")

VALID_PATH = Path.joinpath(ASSETS_PATH, "out_valid")

DETECTION_PATH = Path.joinpath(ASSETS_PATH, "detection")
IMAGES_IN_PATH = Path.joinpath(DETECTION_PATH, "images_in")
IMAGES_OUT_PATH = Path.joinpath(DETECTION_PATH, "images_out")
VIDEOS_IN_PATH = Path.joinpath(DETECTION_PATH, "videos_in")
VIDEOS_OUT_PATH = Path.joinpath(DETECTION_PATH, "videos_out")

DATESET_PATH = Path.joinpath(ASSETS_PATH, "dataset")
GREEN_PATH = Path.joinpath(DATESET_PATH, "0_green")
YELLOW_PATH = Path.joinpath(DATESET_PATH, "1_yellow")
RED_PATH = Path.joinpath(DATESET_PATH, "2_red")
NOT_PATH = Path.joinpath(DATESET_PATH, "3_not")

EXTRACTION_PATH = Path.joinpath(ASSETS_PATH, "extraction")
CROPPED_IMAGES_PATH = Path.joinpath(EXTRACTION_PATH, "cropped")
INPUT_PATH = Path.joinpath(EXTRACTION_PATH, "input")


PATHS = (LOGS_PATH, ASSETS_PATH, VALID_PATH, DETECTION_PATH, IMAGES_IN_PATH, IMAGES_OUT_PATH, VIDEOS_IN_PATH, VIDEOS_OUT_PATH,
         DATESET_PATH, GREEN_PATH, YELLOW_PATH, RED_PATH, NOT_PATH, EXTRACTION_PATH, CROPPED_IMAGES_PATH, INPUT_PATH)

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.FileHandler(str(Path.joinpath(LOGS_PATH, f"{__name__}.log")))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def create_dirs(fresh: bool = False) -> None:
    if fresh:
        rmtree(ASSETS_PATH)
        rmtree(LOGS_PATH)
        Path.unlink(MODEL_PATH)
        create_dirs()
        logger.info("Deleted all dirs")
    else:
        for path in PATHS:
            if not Path.exists(path):
                Path.mkdir(path)
                logger.info(f"Created dir {path}")
