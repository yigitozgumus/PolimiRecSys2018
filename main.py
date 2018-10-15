from project.project import RecommenderSystem
from project.utils.config import process_config


def main():
    config = process_config("configs/initial_config.json")
    recSys = RecommenderSystem(config)


if __name__ == "__main__":
    main()
