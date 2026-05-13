// SPDX-License-Identifier: CeCILL-2.1
// High-level JavaScript / TypeScript wrapper around the Emscripten module.
// Provides Float64Array-friendly fit/predict that mirrors the C++ public API.
//
// Usage (Node.js):
//     import { AOMPLS } from './aompls.mjs';
//     const lib = await AOMPLS.load();
//     const model = lib.fit(X, y, { max_components: 15, preproc: 'snv' });
//     const pred = lib.predict(model, Xnew);
//
// Usage (browser): same API, served as ES module.

import createModule from '../dist/aompls.mjs';  // Emscripten-generated loader

export class AOMPLS {
    constructor(module) {
        this._m = module;
    }

    static async load() {
        const module = await createModule();
        return new AOMPLS(module);
    }

    /**
     * Fit AOM-PLS (compact, PLS1).
     * @param {Float64Array|number[][]} X Row-major (n x p) data. Either a flat
     *     Float64Array of length n*p, or a nested array of rows.
     * @param {Float64Array|number[]} y Length-n response.
     * @param {object} [opts] Optional overrides; merged on top of defaults.
     */
    fit(X, y, opts = {}) {
        const { flat, n, p } = this._toFlat(X);
        const y_flat = this._toFloat64(y);
        if (y_flat.length !== n) throw new Error(`y length (${y_flat.length}) != n (${n})`);
        const cfg = this._m.defaultConfig();
        for (const [k, v] of Object.entries(opts)) {
            if (!(k in cfg)) {
                // Emscripten value_object accepts only declared fields.
                throw new Error(`Unknown option: ${k}`);
            }
            cfg[k] = v;
        }
        return this._m.fit(this._toVector(flat), n, p, this._toVector(y_flat), cfg);
    }

    predict(model, X) {
        const { flat, n, p } = this._toFlat(X);
        if (p !== model.n_features)
            throw new Error(`X has ${p} features, model expects ${model.n_features}`);
        const out = this._m.predict(model, this._toVector(flat), n);
        return this._fromVector(out);
    }

    _toFlat(X) {
        if (X instanceof Float64Array) {
            // We don't know p; require user to pass {flat, n, p} for raw mode.
            throw new Error("X as Float64Array requires {flat, n, p} object or a nested array");
        }
        if (typeof X === 'object' && X.flat instanceof Float64Array) {
            return { flat: X.flat, n: X.n, p: X.p };
        }
        if (Array.isArray(X)) {
            const n = X.length;
            const p = X[0]?.length || 0;
            const flat = new Float64Array(n * p);
            for (let i = 0; i < n; ++i)
                for (let j = 0; j < p; ++j) flat[i * p + j] = X[i][j];
            return { flat, n, p };
        }
        throw new Error("X must be a 2D array or {flat, n, p}");
    }

    _toFloat64(v) {
        if (v instanceof Float64Array) return v;
        return new Float64Array(v);
    }

    _toVector(arr) {
        const vec = new this._m.VectorDouble();
        for (const x of arr) vec.push_back(x);
        return vec;
    }

    _fromVector(vec) {
        const sz = vec.size();
        const out = new Float64Array(sz);
        for (let i = 0; i < sz; ++i) out[i] = vec.get(i);
        return out;
    }
}
