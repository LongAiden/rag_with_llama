"""
Entity and relationship types for Machine Learning/Deep Learning domain.
"""

from enum import Enum


class EntityType(str, Enum):
    """Entity types specific to ML/DL domain."""

    # Core ML Concepts
    ALGORITHM = "ALGORITHM"              # Neural networks, gradient descent, etc.
    MODEL = "MODEL"                      # ResNet, BERT, GPT, etc.
    ARCHITECTURE = "ARCHITECTURE"        # Transformer, CNN, RNN, etc.
    TECHNIQUE = "TECHNIQUE"              # Regularization, normalization, etc.
    CONCEPT = "CONCEPT"                  # Overfitting, bias-variance, etc.

    # Model Components
    LAYER = "LAYER"                      # Conv layer, attention layer, etc.
    ACTIVATION = "ACTIVATION"            # ReLU, Sigmoid, etc.
    LOSS_FUNCTION = "LOSS_FUNCTION"      # Cross-entropy, MSE, etc.
    OPTIMIZER = "OPTIMIZER"              # Adam, SGD, RMSprop, etc.

    # Data Related
    DATASET = "DATASET"                  # ImageNet, COCO, etc.
    DATA_TYPE = "DATA_TYPE"              # Image, text, audio, etc.
    PREPROCESSING = "PREPROCESSING"      # Normalization, augmentation, etc.
    FEATURE = "FEATURE"                  # Image features, text embeddings, etc.

    # Performance & Metrics
    METRIC = "METRIC"                    # Accuracy, F1-score, BLEU, etc.
    BENCHMARK = "BENCHMARK"              # Performance benchmarks
    EVALUATION = "EVALUATION"            # Evaluation methods

    # Tasks & Applications
    TASK = "TASK"                        # Classification, detection, etc.
    APPLICATION = "APPLICATION"          # Computer vision, NLP, etc.
    DOMAIN = "DOMAIN"                    # Healthcare AI, autonomous driving, etc.

    # Frameworks & Tools
    FRAMEWORK = "FRAMEWORK"              # PyTorch, TensorFlow, etc.
    LIBRARY = "LIBRARY"                  # NumPy, scikit-learn, etc.
    TOOL = "TOOL"                        # Jupyter, MLflow, etc.

    # Research & Theory
    PAPER = "PAPER"                      # Research papers
    RESEARCHER = "RESEARCHER"            # Authors, contributors
    ORGANIZATION = "ORGANIZATION"        # Research labs, companies
    THEORY = "THEORY"                    # Theoretical concepts

    # Training & Deployment
    HYPERPARAMETER = "HYPERPARAMETER"    # Learning rate, batch size, etc.
    TRAINING_METHOD = "TRAINING_METHOD"  # Transfer learning, fine-tuning, etc.
    HARDWARE = "HARDWARE"                # GPU, TPU, etc.
    DEPLOYMENT = "DEPLOYMENT"            # Edge deployment, cloud, etc.

    # Other
    CHALLENGE = "CHALLENGE"              # Problems to solve
    LIMITATION = "LIMITATION"            # Known limitations
    GENERAL = "GENERAL"                  # General entities

    # NLP-Specific
    TOKENIZER = "TOKENIZER"              # BPE, WordPiece, SentencePiece, etc.
    EMBEDDING = "EMBEDDING"              # Word2Vec, GloVe, FastText, etc.
    CORPUS = "CORPUS"                    # Text corpora (Wikipedia, Common Crawl, etc.)
    LANGUAGE = "LANGUAGE"                # Language specifications (English, multilingual, etc.)
    LINGUISTIC_FEATURE = "LINGUISTIC_FEATURE"  # Syntax, semantics, morphology, POS tags, etc.
    NLP_COMPONENT = "NLP_COMPONENT"      # Parsers, taggers, chunkers, etc.

    # LLM-Specific
    PROMPT_TEMPLATE = "PROMPT_TEMPLATE"  # Prompt patterns, few-shot examples, chain-of-thought, etc.
    FINE_TUNING_METHOD = "FINE_TUNING_METHOD"  # LoRA, QLoRA, Prefix tuning, P-tuning, Adapter layers, etc.
    QUANTIZATION = "QUANTIZATION"        # INT8, INT4, GPTQ, AWQ, bitsandbytes, etc.
    ALIGNMENT_METHOD = "ALIGNMENT_METHOD"  # RLHF, DPO, PPO, RLAIF, Constitutional AI, etc.
    DECODING_STRATEGY = "DECODING_STRATEGY"  # Beam search, top-k, top-p, nucleus sampling, temperature, etc.
    CONTEXT_WINDOW = "CONTEXT_WINDOW"    # Context length specs (2k, 4k, 128k tokens, etc.)
    ATTENTION_MECHANISM = "ATTENTION_MECHANISM"  # Self-attention, cross-attention, multi-query, flash attention, etc.
    POSITION_ENCODING = "POSITION_ENCODING"  # Absolute, relative, RoPE, ALiBi, etc.
    INFERENCE_ENGINE = "INFERENCE_ENGINE"  # vLLM, TGI, llama.cpp, Ollama, etc.
    SAFETY_TECHNIQUE = "SAFETY_TECHNIQUE"  # Content filtering, guardrails, red teaming methods, etc.


class RelationshipType(str, Enum):
    """Relationship types for ML/DL knowledge graph."""

    # Hierarchical relationships
    IS_A = "IS_A"                        # ResNet IS_A CNN architecture
    PART_OF = "PART_OF"                  # Attention layer PART_OF Transformer
    SUBTYPE_OF = "SUBTYPE_OF"           # BERT SUBTYPE_OF Transformer
    EXTENDS = "EXTENDS"                  # GPT-3 EXTENDS GPT-2

    # Usage relationships
    USES = "USES"                        # Model USES optimizer
    REQUIRES = "REQUIRES"                # Training REQUIRES dataset
    APPLIES_TO = "APPLIES_TO"           # Technique APPLIES_TO task
    IMPLEMENTS = "IMPLEMENTS"            # Code IMPLEMENTS algorithm

    # Performance relationships
    IMPROVES = "IMPROVES"                # Technique IMPROVES metric
    OUTPERFORMS = "OUTPERFORMS"         # Model A OUTPERFORMS Model B
    EVALUATED_ON = "EVALUATED_ON"        # Model EVALUATED_ON benchmark
    ACHIEVES = "ACHIEVES"                # Model ACHIEVES metric score

    # Development relationships
    TRAINED_ON = "TRAINED_ON"            # Model TRAINED_ON dataset
    OPTIMIZED_BY = "OPTIMIZED_BY"       # Training OPTIMIZED_BY optimizer
    DEVELOPED_BY = "DEVELOPED_BY"        # Model DEVELOPED_BY organization
    PUBLISHED_IN = "PUBLISHED_IN"        # Research PUBLISHED_IN paper

    # Problem-solution relationships
    SOLVES = "SOLVES"                    # Algorithm SOLVES problem
    ADDRESSES = "ADDRESSES"              # Technique ADDRESSES challenge
    MITIGATES = "MITIGATES"             # Method MITIGATES limitation

    # Similarity & comparison
    SIMILAR_TO = "SIMILAR_TO"            # Concept A SIMILAR_TO Concept B
    ALTERNATIVE_TO = "ALTERNATIVE_TO"    # Method A ALTERNATIVE_TO Method B
    COMPETES_WITH = "COMPETES_WITH"     # Model A COMPETES_WITH Model B

    # Dependency relationships
    DEPENDS_ON = "DEPENDS_ON"            # Feature DEPENDS_ON preprocessing
    ENABLES = "ENABLES"                  # Hardware ENABLES training
    BASED_ON = "BASED_ON"                # Work BASED_ON prior research

    # Temporal relationships
    PRECEDES = "PRECEDES"                # Paper A PRECEDES Paper B
    SUCCEEDS = "SUCCEEDS"                # Version 2 SUCCEEDS Version 1
    INSPIRED_BY = "INSPIRED_BY"         # Work INSPIRED_BY prior work

    # Application relationships
    APPLIED_IN = "APPLIED_IN"            # Model APPLIED_IN domain
    SUITABLE_FOR = "SUITABLE_FOR"        # Architecture SUITABLE_FOR task

    # General relationships
    RELATED_TO = "RELATED_TO"            # Generic relationship
    MENTIONED_WITH = "MENTIONED_WITH"    # Co-occurrence in text

    # NLP/LLM-Specific relationships
    TOKENIZED_BY = "TOKENIZED_BY"        # Text TOKENIZED_BY tokenizer
    EMBEDDED_BY = "EMBEDDED_BY"          # Token EMBEDDED_BY embedding model
    PRETRAINED_ON = "PRETRAINED_ON"      # Model PRETRAINED_ON corpus
    FINE_TUNED_WITH = "FINE_TUNED_WITH"  # Model FINE_TUNED_WITH method
    QUANTIZED_TO = "QUANTIZED_TO"        # Model QUANTIZED_TO format
    ALIGNED_WITH = "ALIGNED_WITH"        # Model ALIGNED_WITH alignment method
    SUPPORTS_LANGUAGE = "SUPPORTS_LANGUAGE"  # Model SUPPORTS_LANGUAGE language
    GENERATES = "GENERATES"              # Model GENERATES output type
    PROMPTED_WITH = "PROMPTED_WITH"      # Model PROMPTED_WITH prompt template
    DECODED_WITH = "DECODED_WITH"        # Output DECODED_WITH strategy
    SERVED_BY = "SERVED_BY"              # Model SERVED_BY inference engine


# Entity type descriptions for LLM extraction
ENTITY_TYPE_DESCRIPTIONS = {
    # Core ML Concepts
    EntityType.ALGORITHM: "A computational procedure or formula (e.g., backpropagation, gradient descent)",
    EntityType.MODEL: "A specific ML/DL model instance (e.g., BERT, GPT-4, ResNet-50, LLaMA)",
    EntityType.ARCHITECTURE: "A model structure or design pattern (e.g., Transformer, CNN, LSTM, Encoder-Decoder)",
    EntityType.TECHNIQUE: "A method or approach (e.g., dropout, batch normalization, data augmentation)",
    EntityType.CONCEPT: "An ML/DL concept or principle (e.g., overfitting, bias-variance tradeoff, generalization)",

    # Model Components
    EntityType.LAYER: "A neural network layer (e.g., convolutional layer, attention layer, feedforward layer)",
    EntityType.ACTIVATION: "An activation function (e.g., ReLU, GELU, Sigmoid, Tanh)",
    EntityType.LOSS_FUNCTION: "A loss or objective function (e.g., cross-entropy, MSE, contrastive loss)",
    EntityType.OPTIMIZER: "An optimization algorithm (e.g., Adam, SGD, AdamW, RMSprop)",

    # Data Related
    EntityType.DATASET: "A collection of data used for training/evaluation (e.g., ImageNet, MNIST, GLUE)",
    EntityType.DATA_TYPE: "A type of data (e.g., image, text, audio, video, multimodal)",
    EntityType.PREPROCESSING: "Data preprocessing method (e.g., normalization, augmentation, cleaning)",
    EntityType.FEATURE: "A feature or representation (e.g., image features, text embeddings, hidden states)",

    # Performance & Metrics
    EntityType.METRIC: "A performance measurement (e.g., accuracy, F1-score, perplexity, BLEU, ROUGE)",
    EntityType.BENCHMARK: "A standardized evaluation benchmark (e.g., SuperGLUE, ImageNet, MMLU)",
    EntityType.EVALUATION: "An evaluation method or protocol (e.g., cross-validation, human evaluation)",

    # Tasks & Applications
    EntityType.TASK: "An ML problem type (e.g., classification, detection, translation, text generation)",
    EntityType.APPLICATION: "An application domain (e.g., computer vision, NLP, speech recognition)",
    EntityType.DOMAIN: "A specific application domain (e.g., healthcare AI, autonomous driving, finance)",

    # Frameworks & Tools
    EntityType.FRAMEWORK: "A software framework (e.g., PyTorch, TensorFlow, JAX, Hugging Face)",
    EntityType.LIBRARY: "A software library (e.g., NumPy, scikit-learn, transformers, LangChain)",
    EntityType.TOOL: "A development or deployment tool (e.g., Jupyter, MLflow, Weights & Biases)",

    # Research & Theory
    EntityType.PAPER: "A research paper or publication (e.g., 'Attention Is All You Need')",
    EntityType.RESEARCHER: "A researcher or author (e.g., Geoffrey Hinton, Yann LeCun)",
    EntityType.ORGANIZATION: "A research lab or company (e.g., OpenAI, Google DeepMind, Meta AI)",
    EntityType.THEORY: "A theoretical concept or principle (e.g., information theory, universal approximation)",

    # Training & Deployment
    EntityType.HYPERPARAMETER: "A training hyperparameter (e.g., learning rate, batch size, warmup steps)",
    EntityType.TRAINING_METHOD: "A training approach (e.g., transfer learning, self-supervised learning, curriculum learning)",
    EntityType.HARDWARE: "Hardware for computation (e.g., GPU, TPU, A100, H100)",
    EntityType.DEPLOYMENT: "A deployment method or platform (e.g., edge deployment, cloud API, on-premise)",

    # Other
    EntityType.CHALLENGE: "A problem or challenge in ML/DL (e.g., catastrophic forgetting, mode collapse)",
    EntityType.LIMITATION: "A known limitation or constraint (e.g., context length limit, computational cost)",
    EntityType.GENERAL: "A general entity not fitting other categories",

    # NLP-Specific
    EntityType.TOKENIZER: "A text tokenization method (e.g., BPE, WordPiece, SentencePiece, tiktoken)",
    EntityType.EMBEDDING: "A word or token embedding method (e.g., Word2Vec, GloVe, FastText, learned embeddings)",
    EntityType.CORPUS: "A text corpus or dataset (e.g., Wikipedia, Common Crawl, BookCorpus, The Pile)",
    EntityType.LANGUAGE: "A natural language or language family (e.g., English, multilingual, low-resource languages)",
    EntityType.LINGUISTIC_FEATURE: "A linguistic concept or feature (e.g., syntax, semantics, morphology, POS tags)",
    EntityType.NLP_COMPONENT: "An NLP component or module (e.g., parser, tagger, NER system, sentiment analyzer)",

    # LLM-Specific
    EntityType.PROMPT_TEMPLATE: "A prompt pattern or template (e.g., few-shot prompts, chain-of-thought, system prompts)",
    EntityType.FINE_TUNING_METHOD: "A parameter-efficient fine-tuning method (e.g., LoRA, QLoRA, Prefix tuning, Adapter layers)",
    EntityType.QUANTIZATION: "A model quantization technique (e.g., INT8, INT4, GPTQ, AWQ, bitsandbytes)",
    EntityType.ALIGNMENT_METHOD: "An AI alignment technique (e.g., RLHF, DPO, PPO, RLAIF, Constitutional AI)",
    EntityType.DECODING_STRATEGY: "A text generation decoding method (e.g., beam search, top-k, top-p, nucleus sampling)",
    EntityType.CONTEXT_WINDOW: "A context length specification (e.g., 2k tokens, 4k tokens, 128k tokens, infinite context)",
    EntityType.ATTENTION_MECHANISM: "An attention mechanism variant (e.g., self-attention, cross-attention, multi-query, flash attention)",
    EntityType.POSITION_ENCODING: "A positional encoding method (e.g., absolute position, relative position, RoPE, ALiBi)",
    EntityType.INFERENCE_ENGINE: "An LLM inference engine or runtime (e.g., vLLM, TGI, llama.cpp, Ollama, TensorRT-LLM)",
    EntityType.SAFETY_TECHNIQUE: "An AI safety or alignment technique (e.g., content filtering, guardrails, red teaming)",
}

# Relationship type descriptions for LLM extraction
RELATIONSHIP_TYPE_DESCRIPTIONS = {
    # Hierarchical relationships
    RelationshipType.IS_A: "Entity is a type/instance of another (e.g., BERT IS_A Transformer model)",
    RelationshipType.PART_OF: "Entity is a component of another (e.g., Attention layer PART_OF Transformer)",
    RelationshipType.SUBTYPE_OF: "Entity is a specialized version of another (e.g., GPT-4 SUBTYPE_OF GPT)",
    RelationshipType.EXTENDS: "Entity extends or builds upon another (e.g., GPT-3 EXTENDS GPT-2)",

    # Usage relationships
    RelationshipType.USES: "Entity uses or employs another entity (e.g., Model USES optimizer)",
    RelationshipType.REQUIRES: "Entity requires another to function (e.g., Training REQUIRES dataset)",
    RelationshipType.APPLIES_TO: "Entity applies to another (e.g., Technique APPLIES_TO task)",
    RelationshipType.IMPLEMENTS: "Entity implements another (e.g., Code IMPLEMENTS algorithm)",

    # Performance relationships
    RelationshipType.IMPROVES: "Entity improves or enhances another (e.g., LoRA IMPROVES fine-tuning efficiency)",
    RelationshipType.OUTPERFORMS: "Entity performs better than another (e.g., GPT-4 OUTPERFORMS GPT-3)",
    RelationshipType.EVALUATED_ON: "Entity is evaluated on a benchmark/dataset (e.g., Model EVALUATED_ON MMLU)",
    RelationshipType.ACHIEVES: "Entity achieves a specific metric score (e.g., Model ACHIEVES 95% accuracy)",

    # Development relationships
    RelationshipType.TRAINED_ON: "Model is trained on a dataset (e.g., BERT TRAINED_ON BookCorpus)",
    RelationshipType.OPTIMIZED_BY: "Training is optimized by an optimizer (e.g., Training OPTIMIZED_BY Adam)",
    RelationshipType.DEVELOPED_BY: "Entity is developed by an organization/researcher (e.g., GPT DEVELOPED_BY OpenAI)",
    RelationshipType.PUBLISHED_IN: "Research is published in a paper/venue (e.g., Transformer PUBLISHED_IN 'Attention Is All You Need')",

    # Problem-solution relationships
    RelationshipType.SOLVES: "Entity solves a specific problem (e.g., Attention SOLVES long-range dependencies)",
    RelationshipType.ADDRESSES: "Entity addresses a challenge (e.g., Dropout ADDRESSES overfitting)",
    RelationshipType.MITIGATES: "Entity mitigates a limitation (e.g., Gradient clipping MITIGATES exploding gradients)",

    # Similarity & comparison
    RelationshipType.SIMILAR_TO: "Entity is similar to another (e.g., BERT SIMILAR_TO RoBERTa)",
    RelationshipType.ALTERNATIVE_TO: "Entity is an alternative to another (e.g., JAX ALTERNATIVE_TO PyTorch)",
    RelationshipType.COMPETES_WITH: "Entity competes with another (e.g., Gemini COMPETES_WITH GPT-4)",

    # Dependency relationships
    RelationshipType.DEPENDS_ON: "Entity depends on another (e.g., Fine-tuning DEPENDS_ON pretrained model)",
    RelationshipType.ENABLES: "Entity enables another (e.g., GPU ENABLES large-scale training)",
    RelationshipType.BASED_ON: "Entity is built upon or derived from another (e.g., BERT BASED_ON Transformer)",

    # Temporal relationships
    RelationshipType.PRECEDES: "Entity came before another (e.g., Word2Vec PRECEDES BERT)",
    RelationshipType.SUCCEEDS: "Entity came after another (e.g., GPT-4 SUCCEEDS GPT-3)",
    RelationshipType.INSPIRED_BY: "Entity was inspired by another (e.g., T5 INSPIRED_BY BERT)",

    # Application relationships
    RelationshipType.APPLIED_IN: "Entity is applied in a domain (e.g., YOLO APPLIED_IN computer vision)",
    RelationshipType.SUITABLE_FOR: "Entity is suitable for a task (e.g., Transformer SUITABLE_FOR sequence tasks)",

    # General relationships
    RelationshipType.RELATED_TO: "Generic relationship between entities",
    RelationshipType.MENTIONED_WITH: "Entities co-occur in text",

    # NLP/LLM-Specific relationships
    RelationshipType.TOKENIZED_BY: "Text is tokenized by a tokenizer (e.g., Input TOKENIZED_BY BPE)",
    RelationshipType.EMBEDDED_BY: "Token is embedded by an embedding model (e.g., Word EMBEDDED_BY Word2Vec)",
    RelationshipType.PRETRAINED_ON: "Model is pretrained on a corpus (e.g., GPT-3 PRETRAINED_ON Common Crawl)",
    RelationshipType.FINE_TUNED_WITH: "Model is fine-tuned with a method (e.g., LLaMA FINE_TUNED_WITH LoRA)",
    RelationshipType.QUANTIZED_TO: "Model is quantized to a format (e.g., LLaMA QUANTIZED_TO INT4)",
    RelationshipType.ALIGNED_WITH: "Model is aligned with a method (e.g., GPT-4 ALIGNED_WITH RLHF)",
    RelationshipType.SUPPORTS_LANGUAGE: "Model supports a language (e.g., mBERT SUPPORTS_LANGUAGE multilingual)",
    RelationshipType.GENERATES: "Model generates output type (e.g., GPT GENERATES text)",
    RelationshipType.PROMPTED_WITH: "Model is prompted with a template (e.g., GPT-4 PROMPTED_WITH few-shot)",
    RelationshipType.DECODED_WITH: "Output is decoded with a strategy (e.g., Text DECODED_WITH beam search)",
    RelationshipType.SERVED_BY: "Model is served by an inference engine (e.g., LLaMA SERVED_BY vLLM)",
}
