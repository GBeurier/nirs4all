// Parity tests against the JSON fixtures (run after `npm run build`).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { AOMPLS } from '../src/aompls.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REF_DIR = path.resolve(HERE, '..', '..', 'cpp', 'tests', 'reference');
const datasets = ['BEER', 'CORN', 'ALPINE'];
const cases = [
    { name: 'kfold5',       one_se_rule: false },
    { name: 'kfold5_oneSE', one_se_rule: true  },
    { name: 'spxy5',        one_se_rule: false },
];

const lib = await AOMPLS.load();

let fails = 0;
for (const ds of datasets) {
    const refPath = path.join(REF_DIR, `${ds}.json`);
    if (!fs.existsSync(refPath)) { console.warn(`missing ${refPath}`); continue; }
    const raw = JSON.parse(fs.readFileSync(refPath, 'utf8'));
    for (const cs of cases) {
        const ref = raw[cs.name];
        const X = ref.X;  // nested array
        const y = ref.y;
        const folds = ref.fold_test_indices.map(f => f.map(i => i | 0));
        const model = lib.fit(X, y, {
            max_components: ref.max_components,
            cv_mode: 'external',
            external_folds: folds,
            one_se_rule: cs.one_se_rule,
            random_state: 0,
        });
        const tag = `[${ds}/${cs.name}]`;
        if (model.selected_operator_name !== ref.selected_operator_name) {
            console.error(`FAIL ${tag} op (js=${model.selected_operator_name} ref=${ref.selected_operator_name})`);
            fails++; continue;
        }
        if (model.n_components_selected !== ref.n_components_selected) {
            console.error(`FAIL ${tag} k (js=${model.n_components_selected} ref=${ref.n_components_selected})`);
            fails++; continue;
        }
        let coefDiff = 0;
        for (let i = 0; i < ref.coef.length; ++i)
            coefDiff = Math.max(coefDiff, Math.abs(model.coef[i] - ref.coef[i]));
        if (coefDiff > 1e-8) { console.error(`FAIL ${tag} coef |Δ|=${coefDiff.toExponential(3)}`); fails++; continue; }
        const pred = lib.predict(model, X);
        let predDiff = 0;
        for (let i = 0; i < pred.length; ++i)
            predDiff = Math.max(predDiff, Math.abs(pred[i] - ref.predictions_train[i]));
        if (predDiff > 1e-8) { console.error(`FAIL ${tag} pred |Δ|=${predDiff.toExponential(3)}`); fails++; continue; }
        console.log(`${tag} OK selected=${model.selected_operator_name} k=${model.n_components_selected} coef|Δ|=${coefDiff.toExponential(3)} pred|Δ|=${predDiff.toExponential(3)}`);
    }
}
if (fails === 0) { console.log('parity: ALL PASS'); process.exit(0); }
else { console.error(`parity: ${fails} failure(s)`); process.exit(1); }
