from pathlib import Path
from shutil import rmtree

from loguru import logger


BASE_PATH = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_PATH.joinpath("model.h5")
LOGS_PATH = BASE_PATH.joinpath(".logs")
ASSETS_PATH = BASE_PATH.joinpath("assets")

VALID_PATH = ASSETS_PATH.joinpath("out_valid")

DETECTION_PATH = ASSETS_PATH.joinpath("detection")
IMAGES_IN_PATH = DETECTION_PATH.joinpath("images_in")
IMAGES_OUT_PATH = DETECTION_PATH.joinpath("images_out")
VIDEOS_IN_PATH = DETECTION_PATH.joinpath("videos_in")
VIDEOS_OUT_PATH = DETECTION_PATH.joinpath("videos_out")

DATESET_PATH = ASSETS_PATH.joinpath("dataset")
GREEN_PATH = DATESET_PATH.joinpath("0_green")
YELLOW_PATH = DATESET_PATH.joinpath(DATESET_PATH, "1_yellow")
RED_PATH = DATESET_PATH.joinpath("2_red")
NOT_PATH = DATESET_PATH.joinpath("3_not")

EXTRACTION_PATH = ASSETS_PATH.joinpath("extraction")
CROPPED_IMAGES_PATH = EXTRACTION_PATH.joinpath("cropped")
INPUT_PATH = EXTRACTION_PATH.joinpath("input")


PATHS = (LOGS_PATH, VALID_PATH, IMAGES_IN_PATH, IMAGES_OUT_PATH, VIDEOS_IN_PATH, VIDEOS_OUT_PATH,
         GREEN_PATH, YELLOW_PATH, RED_PATH, NOT_PATH, CROPPED_IMAGES_PATH, INPUT_PATH)


@logger.catch
def create_dirs(fresh: bool = False) -> None:
    if fresh:
        rmtree(ASSETS_PATH)
        rmtree(LOGS_PATH)
        MODEL_PATH.unlink()
        create_dirs()
        logger.info("Deleted all dirs")
    else:
        for path in PATHS:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created dir {path}")
