import pytest
from train_script import train

def test_script():
    assert train() == 'success'