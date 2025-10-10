React refactor plan — _read_the_vibe

Goal

- Replace the current vanilla JS + template-driven UI with a small, maintainable React SPA that consumes the existing FastAPI APIs.
- Keep the existing backend API surface unchanged so the migration can be incremental.

High-level approach

1. Deliver a minimal React SPA prototype that reproduces the core Workspace/Datasets functionality and talks to the same endpoints under /api/*.
   - This will be a single-page app served from /webapp/ and will provide a Workspace view and a Predictions view (basic navigation).
2. Iterate: replace each template page (workspace, predictions, pipeline editor) with a React route and component.
3. Add a small client-side state layer (React Context or a lightweight store) for UI state (selected datasets, groups, workspace path) so the UI doesn’t bounce during small updates.
4. Harden UX: incremental updates (optimistic UI) for group add/remove, show toasts for errors, and avoid full-page redraws where possible.
5. Build & serve: in production the React app will be built into static assets and the FastAPI server will serve them at /webapp/ (the server will detect the React bundle dir and prefer it when present).

Why React (brief)

- Component model: easier to reason about the complex UI pieces (file browser, dataset config modal, groups modal, datasets table).
- Declarative code: updates are easier to express via state changes rather than manual DOM manipulation.
- Reusable components and simpler testing.

Environment implications

- Node.js and npm/yarn required for frontend development and building the production bundle.
- CI jobs should add a step to build the React app and archive its output into the package (or serve a build artifact). Typical tasks:
  - npm install
  - npm run build
  - copy build output to /webapp_react (or configure server to point to build dir)
- Local dev: two processes during development are recommended:
  - FastAPI backend (nirs4all --ui) on e.g. http://localhost:8000
  - React dev server (Vite/CRA) on e.g. http://localhost:5173 with CORS or a small proxy to backend APIs

FastAPI + React stack — what it means

- Backend (FastAPI): API-only server that exposes all app logic under /api/* (unchanged). It also continues to serve static frontend files in production.
- Frontend (React SPA): single-page app that consumes the API and implements UI logic, state management, and navigation.
- Deployment considerations:
  - Build step: a React build (static files) must be generated and deployed alongside the Python app; the FastAPI app serves the static files (or they can be served by a CDN/web server).
  - CORS & auth: when running the dev servers separately you'll need to allow the frontend origin in the FastAPI CORS config or use a dev proxy.
  - API compatibility: backend must maintain API contracts during the migration so the React app can be rolled out gradually.

Incremental migration strategy

- Start with a small SPA prototype that reimplements the Linked Datasets page and uses the existing endpoints.
- Replace the template route mounting so that when the React bundle is present the server serves the SPA at /webapp/.
- Iterate: move other pages to React and deprecate the old templates gradually.

Developer workflow

- Frontend development: use Vite + React + TypeScript template (recommended) or CRA. Use Tailwind for styling to keep parity with current UI.
- Backend development: continue running the FastAPI server. During development either run the React dev server and enable CORS or add a proxy to the backend.

Backwards compatibility & testing

- The FastAPI server will detect the presence of the React bundle (webapp_react/) and serve it; if not present the existing ui_templates/ and legacy webapp/ remain available.
- Tests should be updated as pages move to React (some tests validating template presence will be updated to validate SPA responses instead).

Next practical steps (short-term)

1. Add a minimal React SPA scaffold that mounts at /webapp/ and reads datasets from /api/workspace/datasets.
2. Wire the workspace page and a small datasets table with group creation (replacing the current DOM-manipulation version).
3. Update server static mounts to prefer the React bundle when present (implemented).
4. Iterate on UX and replace more pages with React components.

Notes

- The initial React scaffold will use CDN-hosted React for minimal friction (no build required for the prototype). For production ready work we will switch to a proper toolchain (Vite + npm build) and add a CI step to build the frontend.

