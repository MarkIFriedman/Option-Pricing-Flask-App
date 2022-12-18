import pytest
from services.web.app import app
import pandas as pd
import json


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_train_set_params(client):
    response = client.get('/train_set_params').json
    params = response['current generator params']
    assert 'sigma' in params
    assert 'spot' in params
    assert 'test_size' in params
    assert 'time' in params
    assert 'train_size' in params


def test_models_list(client):
    response = client.get('/list_of_models').json
    assert 'Availiable models' in response
    assert type(response['Availiable models']) == type({})
