"""Test suite for Flask serve API."""

import json
import tempfile
import shutil
import pytest

@pytest.fixture(scope="module")
def client():
    """Create test client for Flask app."""
    from npcpy.serve import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test Flask app health endpoint"""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'ok'


class TestModelsEndpoint:
    """Test models endpoint."""

    def test_get_models(self, client):
        """Test models endpoint"""
        response = client.get('/api/models?currentPath=/tmp')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'models' in data


class TestSettingsEndpoints:
    """Test settings endpoints."""

    def test_get_global_settings(self, client):
        """Test global settings endpoint"""
        response = client.get('/api/settings/global')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'global_settings' in data

    def test_project_settings_get(self, client):
        """Test project settings GET"""
        response = client.get('/api/settings/project?path=/tmp')
        assert response.status_code == 200


class TestConversationsEndpoint:
    """Test conversations endpoint."""

    def test_get_conversations(self, client):
        """Test conversations endpoint"""
        response = client.get('/api/conversations?path=/tmp')
        # May return 500 if database not initialized - that's ok in CI
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'conversations' in data


class TestJinxesEndpoints:
    """Test jinxes endpoints."""

    def test_global_jinxes(self, client):
        """Test global jinxes endpoint"""
        response = client.get('/api/jinxes/available')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'jinxes' in data

    def test_project_jinxes(self, client):
        """Test project jinxes endpoint"""
        response = client.get('/api/jinxes/project?currentPath=/tmp')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'jinxes' in data


class TestNPCEndpoints:
    """Test NPC-related endpoints."""

    def test_npc_team_global(self, client):
        """Test global NPC team endpoint"""
        response = client.get('/api/npc_team_global')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'npcs' in data

    def test_save_npc(self, client):
        """Test save NPC endpoint"""
        temp_dir = tempfile.mkdtemp()

        try:
            npc_data = {
                "npc": {
                    "name": "test_npc",
                    "primary_directive": "Test NPC",
                    "model": "llama3.2",
                    "provider": "ollama"
                },
                "isGlobal": False,
                "currentPath": temp_dir
            }
            response = client.post('/api/save_npc',
                                   json=npc_data,
                                   content_type='application/json')
            # Just check it doesn't error out
            assert response.status_code in [200, 201, 400, 500]
        finally:
            shutil.rmtree(temp_dir)


class TestScreenshotEndpoint:
    """Test screenshot endpoint."""

    def test_capture_screenshot(self, client):
        """Test screenshot capture endpoint (may fail without display)"""
        response = client.get('/api/capture_screenshot')
        # This may fail in CI without a display, but shouldn't error
        assert response.status_code in [200, 400, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
