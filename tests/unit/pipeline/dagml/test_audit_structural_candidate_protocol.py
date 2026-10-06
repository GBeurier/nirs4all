"""DAG 0.3.37 objective binding stays strict at the Python candidate boundary."""

from copy import deepcopy

import dag_ml
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import nirs4all
from tests.integration.api.test_structural_hpo_ridge_pls import example


@pytest.mark.parametrize('mutation', ['objective', 'missing_objective', 'extra_field', 'trial', 'recipe', 'catalogue', 'node', 'variant', 'base_choices'])
def test_native_candidate_metadata_tampering_is_refused_before_fit(monkeypatch, tmp_path, mutation):
    original_search = dag_ml.run_host_hpo_search_in_process
    observed = []
    checkpoint_objectives = []

    def intercept(*args, **kwargs):
        factory = kwargs['candidate_callback_factory']
        checkpoint = kwargs['progress_callback']

        def progress(event):
            checkpoint_objectives.append(event['checkpoint']['binding']['objective_fingerprint'])
            return checkpoint(event)

        def candidate(index):
            callback = factory(index)

            def operator(task):
                assert checkpoint_objectives
                modified = deepcopy(task)
                value = modified['variant']['choices']['host_hpo']['value']
                assert value['objective_fingerprint'] == checkpoint_objectives[-1]
                observed.append(index)
                if mutation == 'objective':
                    value['objective_fingerprint'] = '0' * 64
                elif mutation == 'missing_objective':
                    value.pop('objective_fingerprint')
                elif mutation == 'extra_field':
                    value['unbound'] = True
                elif mutation == 'trial':
                    value['trial_index'] += 1
                elif mutation == 'recipe':
                    value['recipe_id'] = 'recipe:unbound'
                elif mutation == 'catalogue':
                    value['catalogue_fingerprint'] = '0' * 64
                elif mutation == 'node':
                    modified['node_plan']['node_id'] = 'model:unbound'
                elif mutation == 'variant':
                    modified['variant_id'] = 'host_hpo:trial:0000000009'
                else:
                    modified['variant']['choices']['unbound'] = {'value': True}
                return callback(modified)

            return operator

        kwargs['candidate_callback_factory'] = candidate
        kwargs['progress_callback'] = progress
        return original_search(*args, **kwargs)

    monkeypatch.setattr(dag_ml, 'run_host_hpo_search_in_process', intercept)
    for estimator in (Ridge, PLSRegression, StandardScaler):
        monkeypatch.setattr(estimator, 'fit', lambda *args, **kwargs: pytest.fail('tampered native task reached fit'))
    with pytest.raises(dag_ml.DagMlRuntimeError, match='ValueError: candidate task native trial, recipe or node disagrees with the structural catalogue'):
        nirs4all.run(
            example.make_pipeline(), example.make_dataset(), tuning=example.make_tuning(tmp_path / 'study'),
            engine='dag-ml', workspace_path=tmp_path / 'workspace', random_state=17,
            refit=True, verbose=0, save_charts=False, save_artifacts=True,
        )
    assert observed == [0]
