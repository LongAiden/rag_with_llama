"""SentenceTransformer embedding generator."""

from typing import List

import logfire
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Generate embeddings using SentenceTransformers."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding generator.

        Args:
            model_name: SentenceTransformer model name
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        Args:
            text: Input text
        Returns:
            List of embedding values
        """
        # Validate and clean input text
        if text is None:
            logfire.warn("Embedding called with None text, using empty string")
            text = ""
        elif not isinstance(text, str):
            logfire.warn("Embedding called with non-string text, converting to string",
                         text_type=type(text).__name__)
            text = str(text)

        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        Args:
            texts: List of input texts
        Returns:
            List of embedding lists
        """
        # Validate and clean input texts - be very defensive
        valid_texts = []
        none_count = 0
        non_string_count = 0
        problematic_items = []

        for i, text in enumerate(texts):
            try:
                if text is None:
                    valid_texts.append("")  # Replace None with empty string
                    none_count += 1
                elif isinstance(text, str):
                    valid_texts.append(text)
                elif isinstance(text, (list, dict, tuple)):
                    # Complex types that can't be directly converted
                    logfire.error("Complex type encountered in embedding batch",
                                  index=i,
                                  text_type=type(text).__name__,
                                  text_value=str(text)[:200])
                    # Fallback to string conversion
                    valid_texts.append(str(text))
                    non_string_count += 1
                    problematic_items.append((i, type(text).__name__))
                else:
                    # Convert non-string to string
                    converted = str(text)
                    valid_texts.append(converted)
                    non_string_count += 1
            except Exception as e:
                logfire.error("Failed to process text at index",
                              index=i,
                              error=str(e),
                              text_type=type(text).__name__)
                valid_texts.append("")  # Use empty string as fallback
                problematic_items.append((i, f"error: {str(e)}"))

        # Log warnings if we encountered invalid texts
        if none_count > 0 or non_string_count > 0:
            logfire.warn("Invalid texts encountered in batch embedding",
                         total_texts=len(texts),
                         none_count=none_count,
                         non_string_count=non_string_count,
                         problematic_items_count=len(problematic_items))

        # Final safety check - ensure all items are strings
        final_texts = []
        for i, text in enumerate(valid_texts):
            if not isinstance(text, str):
                logfire.error("Non-string made it through validation",
                              index=i,
                              type=type(text).__name__)
                final_texts.append(str(text))
            else:
                final_texts.append(text)

        # Ensure we have a proper list of proper strings with aggressive filtering
        safe_texts = []
        for idx, x in enumerate(final_texts):
            try:
                # Convert to string if needed
                if not isinstance(x, str):
                    text = str(x)
                else:
                    text = x

                # Replace empty strings with space
                if len(text.strip()) == 0:
                    text = " "

                # Ensure it's a plain Python str (not bytes, not subclass)
                if isinstance(text, bytes):
                    text = text.decode('utf-8', errors='replace')

                # Final conversion to ensure plain str type
                text = str(text)

                safe_texts.append(text)
            except Exception as e:
                logfire.error(
                    "Failed to process text in final cleanup", index=idx, error=str(e))
                safe_texts.append(" ")  # Fallback to space

        # Verify the conversion worked
        logfire.info("Preparing to encode batch",
                     total_texts=len(safe_texts),
                     all_strings=all(isinstance(t, str) for t in safe_texts),
                     sample_lengths=[len(t) for t in safe_texts[:5]])

        try:
            # Force each text through encode/decode to ensure pure Python strings
            texts_to_encode = []
            for t in safe_texts:
                # Convert to bytes and back to ensure pure string
                clean_text = str(t).encode(
                    'utf-8', errors='replace').decode('utf-8')
                texts_to_encode.append(clean_text)

            # Encode in batches for efficiency (batch_size=32)
            batch_size = 32
            all_embeddings = []

            logfire.info("Starting batch embedding",
                         total_texts=len(texts_to_encode),
                         batch_size=batch_size,
                         num_batches=(len(texts_to_encode) + batch_size - 1) // batch_size)

            for i in range(0, len(texts_to_encode), batch_size):
                batch = texts_to_encode[i:i + batch_size]
                batch_num = i // batch_size + 1

                try:
                    # Encode this batch
                    batch_embeddings = self.model.encode(
                        batch,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        batch_size=len(batch)
                    )
                    all_embeddings.extend([emb.tolist()
                                          for emb in batch_embeddings])

                    if batch_num % 10 == 0:  # Log every 10 batches
                        logfire.info(f"Processed batch {batch_num}",
                                     embeddings_so_far=len(all_embeddings))
                except Exception as batch_error:
                    logfire.error(f"Batch {batch_num} failed, falling back to one-by-one",
                                  error=str(batch_error),
                                  batch_size=len(batch))
                    # Fallback: encode one by one for this batch
                    for text in batch:
                        try:
                            embedding = self.model.encode(
                                text, show_progress_bar=False, convert_to_numpy=True)
                            all_embeddings.append(embedding.tolist())
                        except Exception:
                            all_embeddings.append([0.0] * self.embedding_dim)

            logfire.info("Batch embedding completed",
                         total_texts=len(texts_to_encode),
                         total_embeddings=len(all_embeddings))

            return all_embeddings
        except Exception as e:
            logfire.error("Embedding encoding failed completely",
                          error=str(e),
                          error_type=type(e).__name__,
                          total_texts=len(safe_texts))
            # Log samples for debugging
            for i, text in enumerate(safe_texts[:3]):
                logfire.error(f"Sample text {i}",
                              text_type=type(text).__name__,
                              text_repr=repr(text)[:200],
                              text_len=len(text) if isinstance(text, str) else 0)
            raise
