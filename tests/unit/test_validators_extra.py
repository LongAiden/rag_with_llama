"""
Unit tests for api/validators.py (extra coverage beyond test_validate_table_name.py).

Covers:
- validate_upload_params (chunk_size bounds, content_type)
- require_access_password (env-based auth)
- celery_enabled / celery_upload_enabled / entity_extraction_enabled
"""
import os
import pytest
from unittest.mock import patch
from fastapi import HTTPException

from api.validators import (
    validate_upload_params,
    require_access_password,
    celery_enabled,
    celery_upload_enabled,
    entity_extraction_enabled,
)


class TestValidateUploadParams:
    def test_valid_params(self):
        validate_upload_params(512, "application/pdf")

    def test_chunk_size_too_small(self):
        with pytest.raises(HTTPException) as exc:
            validate_upload_params(64, "application/pdf")
        assert exc.value.status_code == 400

    def test_chunk_size_too_large(self):
        with pytest.raises(HTTPException) as exc:
            validate_upload_params(4096, "application/pdf")
        assert exc.value.status_code == 400

    def test_chunk_size_min_boundary(self):
        validate_upload_params(128, "application/pdf")

    def test_chunk_size_max_boundary(self):
        validate_upload_params(2048, "application/pdf")

    def test_invalid_content_type(self):
        with pytest.raises(HTTPException) as exc:
            validate_upload_params(512, "image/png")
        assert exc.value.status_code == 400

    def test_pdf_content_type(self):
        validate_upload_params(512, "application/pdf")

    def test_docx_content_type(self):
        validate_upload_params(512, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    def test_txt_content_type(self):
        validate_upload_params(512, "text/plain")


class TestRequireAccessPassword:
    def test_no_password_configured_allows_any(self):
        with patch.dict(os.environ, {}, clear=True):
            require_access_password(None)
            require_access_password("anything")

    def test_correct_password_passes(self):
        with patch.dict(os.environ, {"APP_ACCESS_PASSWORD": "secret"}):
            require_access_password("secret")

    def test_wrong_password_raises_403(self):
        with patch.dict(os.environ, {"APP_ACCESS_PASSWORD": "secret"}):
            with pytest.raises(HTTPException) as exc:
                require_access_password("wrong")
            assert exc.value.status_code == 403

    def test_no_password_provided_when_required_raises_403(self):
        with patch.dict(os.environ, {"APP_ACCESS_PASSWORD": "secret"}):
            with pytest.raises(HTTPException) as exc:
                require_access_password(None)
            assert exc.value.status_code == 403

    def test_frontend_password_fallback(self):
        with patch.dict(os.environ, {"FRONTEND_PASSWORD": "frontend_secret"}, clear=True):
            require_access_password("frontend_secret")

    def test_app_password_takes_precedence(self):
        with patch.dict(os.environ, {"APP_ACCESS_PASSWORD": "app", "FRONTEND_PASSWORD": "frontend"}):
            require_access_password("app")
            with pytest.raises(HTTPException):
                require_access_password("frontend")

    def test_empty_password_treated_as_not_configured(self):
        with patch.dict(os.environ, {"APP_ACCESS_PASSWORD": ""}, clear=True):
            require_access_password(None)


class TestCeleryEnabled:
    def test_default_false(self):
        with patch.dict(os.environ, {}, clear=True):
            assert celery_enabled() is False

    def test_true_when_set(self):
        with patch.dict(os.environ, {"USE_CELERY_FOR_EXTRACTION": "true"}):
            assert celery_enabled() is True

    def test_true_case_insensitive(self):
        with patch.dict(os.environ, {"USE_CELERY_FOR_EXTRACTION": "True"}):
            assert celery_enabled() is True

    def test_false_when_set_to_false(self):
        with patch.dict(os.environ, {"USE_CELERY_FOR_EXTRACTION": "false"}):
            assert celery_enabled() is False


class TestCeleryUploadEnabled:
    def test_default_false(self):
        with patch.dict(os.environ, {}, clear=True):
            assert celery_upload_enabled() is False

    def test_true_when_set(self):
        with patch.dict(os.environ, {"USE_CELERY_FOR_UPLOAD": "true"}):
            assert celery_upload_enabled() is True


class TestEntityExtractionEnabled:
    def test_default_true(self):
        with patch.dict(os.environ, {}, clear=True):
            assert entity_extraction_enabled() is True

    def test_false_when_disabled(self):
        with patch.dict(os.environ, {"ENABLE_ENTITY_EXTRACTION": "false"}):
            assert entity_extraction_enabled() is False

    def test_true_when_explicitly_enabled(self):
        with patch.dict(os.environ, {"ENABLE_ENTITY_EXTRACTION": "true"}):
            assert entity_extraction_enabled() is True
