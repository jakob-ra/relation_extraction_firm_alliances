from spacy.cli.project.assets import project_assets
from pathlib import Path
from spacy.cli.project.run import project_run
import spacy

def test_rel_project():
    root = Path(__file__).parent
    project_assets(root)
    project_run(root, "all", capture=True)


if __name__ == "__main__":
    project_run(project_dir=Path.cwd(), subcommand='train_cpu')
