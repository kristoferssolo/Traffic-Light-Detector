from pathlib import Path
from shutil import rmtree

from loguru import logger


BASE_PATH = Path(__file__).resolve().parent.parent
LOGS_PATH = BASE_PATH.joinpath(".logs")
ASSETS_PATH = BASE_PATH.joinpath("assets")
IMAGES_IN_PATH = ASSETS_PATH.joinpath("images_in")
IMAGES_OUT_PATH = ASSETS_PATH.joinpath("images_out")
HAAR_PATH = ASSETS_PATH.joinpath("haar").joinpath("TrafficLights.xml")

PATHS = (LOGS_PATH, IMAGES_IN_PATH, IMAGES_OUT_PATH)


log_level = "DEBUG" if BASE_PATH.joinpath("debug").exists() else "INFO"
# Set up logging
logger.add(LOGS_PATH.joinpath("detection.log"), format="{time} | {level} | {message}", level=log_level, rotation="10 MB", compression="zip")


@logger.catch
def create_dirs(fresh: bool = False) -> None:
    if fresh:
        rmtree(ASSETS_PATH)
        rmtree(LOGS_PATH)
        create_dirs()
        logger.info("Deleted all dirs")
    else:
        for path in PATHS:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory {path}")
