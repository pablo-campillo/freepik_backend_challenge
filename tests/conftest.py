import pytest
from transformers import AutoProcessor, AutoModelForCausalLM


@pytest.fixture(scope="session")
def processor():
    return AutoProcessor.from_pretrained("microsoft/git-base-textcaps")


@pytest.fixture(scope="session")
def model():
    return AutoModelForCausalLM.from_pretrained("microsoft/git-base-textcaps")