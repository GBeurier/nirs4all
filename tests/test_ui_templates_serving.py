from fastapi.testclient import TestClient
from nirs4all.ui import server

client = TestClient(server.app)


def test_workspace_page_served():
    r = client.get('/webapp/workspace.html')
    assert r.status_code == 200
    # basic sanity check that content looks like the workspace page
    assert 'Workspace Management' in r.text or 'No workspace selected' in r.text
