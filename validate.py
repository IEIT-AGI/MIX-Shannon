import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="eval_vqai.yaml")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    if config.name.upper().startswith('VQAI'):
        from src.pipelines.vqai_validate_pipeline import validate
    else:
        from src.pipelines.validate_pipeline import validate

    # Applies optional utilities
    utils.extras(config)

    # Evaluate model
    return validate(config)


if __name__ == "__main__":
    main()
