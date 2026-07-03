# 06. Ресурсы

## Модели (HuggingFace)
- Backbone (единый для команды): `Qwen/Qwen2.5-7B-Instruct`
- Reflection-LM для GEPA: `Qwen/Qwen2.5-32B-Instruct` (fallback — backbone)
- Smoke-тесты пайплайнов: `Qwen/Qwen2.5-0.5B-Instruct`
- NLI: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
  (запасной лёгкий: `cointegrated/rubert-base-cased-nli-threeway`)
- Эмбеддинги: `intfloat/multilingual-e5-large` (префиксы query:/passage:)

## Библиотеки
- vLLM — docs.vllm.ai (serve, chat completions, logprobs)
- DSPy ≥ 2.6 — dspy.ai, github.com/stanfordnlp/dspy (Optimizers -> GEPA;
  туториал dspy.ai/tutorials/gepa_ai_program — наш случай почти дословно)
- standalone GEPA — github.com/gepa-ai/gepa
- openai (клиент к vLLM), transformers, sentence-transformers, scikit-learn,
  numpy, razdel (русская сегментация предложений), pyyaml, tqdm, pytest

## Референсные реализации (смотреть, не копировать вслепую)
- SelfCheckGPT — github.com/potsawee/selfcheckgpt (вариант SelfCheckNLI)
- Semantic uncertainty — github.com/jlko/semantic_uncertainty (кластеризация
  по двунаправленному entailment; Farquhar et al., Nature 2024)
- Lynx prompt — карточка PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct
  (их боевой PASS/FAIL-промпт как референс формата)

## Статьи
- GEPA: arXiv:2507.19457 — Reflective Prompt Evolution (основа Метода 3)
- DSPy: arXiv:2310.03714
- SelfCheckGPT: arXiv:2303.08896 (основа Метода 6)
- Semantic Uncertainty: arXiv:2302.09664
- Lynx: arXiv:2407.08488 (схема PASS/FAIL-вердиктов, сторона H1)
- RAGAS: arXiv:2309.15217 (формализация faithfulness/answer-relevance)
- Alignment-collapse sampling-методов: arXiv:2603.24124 (почему на aligned-
  моделях сэмплы схлопываются в 1 кластер и consistency слепнет — наш
  анализ share_single_cluster)

## Полный список литературы проекта
Overleaf кураторов (ссылка в проектной документации команды).
