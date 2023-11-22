from pathlib import Path

import pytest
from transformers import AutoProcessor, AutoModelForCausalLM

root_dir = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def processor():
    # return AutoProcessor.from_pretrained("microsoft/git-base-textcaps")
    return AutoProcessor.from_pretrained(root_dir / "git-base-textcaps", local_files_only=True)


@pytest.fixture(scope="session")
def model():
    # return AutoModelForCausalLM.from_pretrained("microsoft/git-base-textcaps")
    return AutoModelForCausalLM.from_pretrained(root_dir / "git-base-textcaps", local_files_only=True)