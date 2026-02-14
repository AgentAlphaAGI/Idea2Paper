import importlib.util
import os
import pathlib
import sys
import types
import unittest
from unittest.mock import patch


def _install_stub_modules():
    idea2paper_pkg = types.ModuleType("idea2paper")
    idea2paper_pkg.__path__ = []
    infra_pkg = types.ModuleType("idea2paper.infra")
    infra_pkg.__path__ = []
    providers_pkg = types.ModuleType("idea2paper.infra.llm_providers")
    providers_pkg.__path__ = []

    config_mod = types.ModuleType("idea2paper.config")
    config_mod.EMBEDDING_API_URL = "https://embeddings.example.test/v1/embeddings"
    config_mod.EMBEDDING_MODEL = "text-embedding-3-large"
    config_mod.LLM_API_URL = ""
    config_mod.LLM_BASE_URL = "https://api.openai.com/v1"
    config_mod.LLM_MODEL = "gpt-4o-mini"
    config_mod.LLM_PROVIDER = "openai_compatible_chat"
    config_mod.LLM_ANTHROPIC_VERSION = "2023-06-01"
    config_mod.LLM_EXTRA_BODY = None
    config_mod.LLM_EXTRA_HEADERS = None
    config_mod.NOVELTY_INDEX_DIR = pathlib.Path(".")

    class _PipelineConfig:
        RECALL_USE_OFFLINE_INDEX = False
        RECALL_INDEX_DIR = pathlib.Path(".")

    config_mod.PipelineConfig = _PipelineConfig

    def _dummy_llm_call(*args, **kwargs):
        return {"ok": True, "text": "pong", "error": ""}

    anthropic_mod = types.ModuleType("idea2paper.infra.llm_providers.anthropic")
    anthropic_mod.call_anthropic = _dummy_llm_call
    gemini_mod = types.ModuleType("idea2paper.infra.llm_providers.gemini")
    gemini_mod.call_gemini = _dummy_llm_call
    openai_compatible_mod = types.ModuleType("idea2paper.infra.llm_providers.openai_compatible")
    openai_compatible_mod.call_openai_compatible_chat = _dummy_llm_call
    openai_responses_mod = types.ModuleType("idea2paper.infra.llm_providers.openai_responses")
    openai_responses_mod.call_openai_responses = _dummy_llm_call

    common_mod = types.ModuleType("idea2paper.infra.llm_providers.common")
    common_mod.parse_extra = lambda value: ({}, "")
    common_mod.redact_mapping = lambda mapping: mapping

    run_context_mod = types.ModuleType("idea2paper.infra.run_context")
    run_context_mod.get_logger = lambda: None

    numpy_mod = types.ModuleType("numpy")
    numpy_mod.load = lambda *args, **kwargs: None

    requests_mod = types.ModuleType("requests")
    requests_mod.post = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("not patched"))

    stubs = {
        "idea2paper": idea2paper_pkg,
        "idea2paper.infra": infra_pkg,
        "idea2paper.infra.llm_providers": providers_pkg,
        "idea2paper.config": config_mod,
        "idea2paper.infra.llm_providers.anthropic": anthropic_mod,
        "idea2paper.infra.llm_providers.gemini": gemini_mod,
        "idea2paper.infra.llm_providers.openai_compatible": openai_compatible_mod,
        "idea2paper.infra.llm_providers.openai_responses": openai_responses_mod,
        "idea2paper.infra.llm_providers.common": common_mod,
        "idea2paper.infra.run_context": run_context_mod,
        "numpy": numpy_mod,
        "requests": requests_mod,
    }
    sys.modules.update(stubs)


def _load_startup_preflight_module():
    _install_stub_modules()
    module_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "idea2paper" / "infra" / "startup_preflight.py"
    spec = importlib.util.spec_from_file_location("startup_preflight_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class StartupPreflightEmbeddingKeyFallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_startup_preflight_module()

    def test_embedding_ping_falls_back_to_llm_api_key(self):
        captured = {}

        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        def _fake_post(url, headers, json, timeout):
            captured["url"] = url
            captured["auth"] = headers.get("Authorization")
            captured["timeout"] = timeout
            captured["payload"] = json
            return _FakeResponse()

        with patch.dict(os.environ, {"EMBEDDING_API_KEY": "", "LLM_API_KEY": "llm_key_only"}, clear=False):
            with patch.object(self.module.requests, "post", side_effect=_fake_post):
                ok, dim, err = self.module._embedding_ping_once(timeout=7)

        self.assertTrue(ok)
        self.assertEqual(dim, 3)
        self.assertEqual(err, "")
        self.assertEqual(captured["auth"], "Bearer llm_key_only")
        self.assertEqual(captured["timeout"], 7)
        self.assertEqual(captured["payload"]["input"], "ping")

    def test_embedding_ping_returns_clear_error_when_both_keys_missing(self):
        with patch.dict(os.environ, {"EMBEDDING_API_KEY": "", "LLM_API_KEY": ""}, clear=False):
            ok, dim, err = self.module._embedding_ping_once(timeout=3)

        self.assertFalse(ok)
        self.assertIsNone(dim)
        self.assertIn("EMBEDDING_API_KEY/LLM_API_KEY", err)


if __name__ == "__main__":
    unittest.main()
