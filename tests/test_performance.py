"""Performance tests for the newsletter generator."""

import pytest
import time
import tracemalloc
import gc
import threading
import queue
from unittest.mock import patch, MagicMock, mock_open
from llm_newsletter_generator.llm_newsletter_generator import NewsletterGenerator


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @pytest.fixture
    def generator(self):
        """Returns a mocked NewsletterGenerator instance for performance testing."""
        with (
            patch(
                "llm_newsletter_generator.llm_newsletter_generator.AutoTokenizer.from_pretrained"
            ) as mock_tokenizer,
            patch("llm_newsletter_generator.llm_newsletter_generator.pipeline") as mock_pipeline,
        ):
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.encode.side_effect = lambda x, add_special_tokens=False: [
                1
            ] * len(x)
            mock_tokenizer.return_value = mock_tokenizer_instance

            # Mock pipeline to return quickly for performance testing
            mock_text_gen = MagicMock()
            mock_text_gen.return_value = [{"generated_text": "prompt + generated content"}]
            mock_pipeline.return_value = mock_text_gen

            yield NewsletterGenerator(feed_url="http://example.com/rss")

    def test_initialization_time(self):
        """Benchmark NewsletterGenerator initialization time."""
        start_time = time.time()

        with (
            patch(
                "llm_newsletter_generator.llm_newsletter_generator." "AutoTokenizer.from_pretrained"
            ),
            patch("llm_newsletter_generator.llm_newsletter_generator.pipeline"),
        ):
            generator = NewsletterGenerator(feed_url="http://example.com/rss")

        initialization_time = time.time() - start_time

        # Initialization should complete within reasonable time (with mocks)
        assert (
            initialization_time < 1.0
        ), f"Initialization took {initialization_time:.2f}s, expected < 1.0s"
        assert generator is not None

    @patch("feedparser.parse")
    def test_feed_parsing_performance(self, mock_parse, generator):
        """Benchmark feed parsing performance with various feed sizes."""
        # Test with different feed sizes
        feed_sizes = [10, 50, 100, 500]

        for size in feed_sizes:
            # Create mock entries
            entries = []
            for i in range(size):
                entries.append(
                    {
                        "title": f"Article {i}",
                        "description": f"Description {i}" * 10,  # Make descriptions longer
                        "link": f"http://example.com/{i}",
                    }
                )

            mock_parse.return_value.entries = entries

            start_time = time.time()
            items = generator.get_items(f"<rss>feed with {size} items</rss>")
            parsing_time = time.time() - start_time

            assert len(items) == size
            # Parsing should be fast even for large feeds
            assert parsing_time < 0.1, (
                f"Parsing {size} items took {parsing_time:.3f}s, " f"expected < 0.1s"
            )

    def test_text_generation_performance(self, generator):
        """Benchmark text generation performance."""
        prompts = [
            "Short prompt",
            "Medium length prompt with more details about the topic",
            "Very long prompt " + "with lots of repeated content " * 20,
        ]

        for prompt in prompts:
            with (
                patch("os.path.exists", return_value=False),  # Force generation, not cache
                patch("builtins.open"),
            ):
                start_time = time.time()
                result = generator.generate_text(prompt)
                generation_time = time.time() - start_time

                assert result is not None
                assert len(result) > 0
                # Text generation should complete quickly with mocked models
                assert (
                    generation_time < 0.1
                ), f"Generation took {generation_time:.3f}s, expected < 0.1s"

    def test_cache_performance(self, generator):
        """Benchmark cache read/write performance."""
        prompt = "Test prompt for cache performance"
        cached_content = "Cached response content"

        # Test cache write performance
        with patch("builtins.open"):
            start_time = time.time()
            # Simulate cache write by calling generate_text with cache miss
            with (
                patch("os.path.exists", return_value=False),
                patch.object(
                    generator,
                    "text_generation",
                    return_value=[{"generated_text": f"{prompt}{cached_content}"}],
                ),
            ):
                generator.generate_text(prompt)
            write_time = time.time() - start_time

        # Test cache read performance
        with (
            patch("os.path.exists", return_value=True),
            patch(
                "builtins.open",
                return_value=MagicMock(
                    __enter__=MagicMock(
                        return_value=MagicMock(read=MagicMock(return_value=cached_content))
                    )
                ),
            ),
        ):
            start_time = time.time()
            result = generator.generate_text(prompt)
            read_time = time.time() - start_time

        assert result == cached_content
        # Cache operations should be very fast
        assert write_time < 0.05, f"Cache write took {write_time:.3f}s, expected < 0.05s"
        assert read_time < 0.01, f"Cache read took {read_time:.3f}s, expected < 0.01s"

    @patch.object(NewsletterGenerator, "generate_prompt")
    @patch.object(NewsletterGenerator, "generate_prompt_for_item")
    @patch.object(NewsletterGenerator, "generate_text")
    @patch("llm_newsletter_generator.llm_newsletter_generator.Progress")
    def test_newsletter_creation_performance(
        self,
        mock_progress,
        mock_generate_text,
        mock_generate_prompt_for_item,
        mock_generate_prompt,
        generator,
    ):
        """Benchmark end-to-end newsletter creation performance."""
        # Setup fast mocks
        mock_generate_text.side_effect = lambda x: f"Generated: {x[:20]}..."
        mock_generate_prompt.side_effect = lambda *args: f"Prompt for {args[0]}"
        mock_generate_prompt_for_item.side_effect = lambda item, topic: f"Item prompt for {item[0]}"

        # Test with different newsletter sizes
        item_counts = [5, 10, 25, 50]

        for count in item_counts:
            items = []
            for i in range(count):
                items.append((f"Article {i}", f"Description {i}", f"http://example.com/{i}"))

            start_time = time.time()
            newsletter = generator.create_newsletter(
                f"Newsletter with {count} items", "Technology", items
            )
            creation_time = time.time() - start_time

            assert newsletter is not None
            assert len(newsletter) > 0
            # Newsletter creation should scale reasonably
            expected_max_time = 0.01 * count  # 10ms per item
            assert creation_time < expected_max_time, (
                f"Creating newsletter with {count} items took "
                f"{creation_time:.3f}s, expected < {expected_max_time:.3f}s"
            )


class TestMemoryUsage:
    """Memory usage tests."""

    @pytest.fixture
    def generator(self):
        """Returns a mocked NewsletterGenerator instance."""
        with (
            patch(
                "llm_newsletter_generator.llm_newsletter_generator." "AutoTokenizer.from_pretrained"
            ),
            patch("llm_newsletter_generator.llm_newsletter_generator.pipeline"),
        ):
            yield NewsletterGenerator(feed_url="http://example.com/rss")

    def get_memory_usage(self):
        """Get current memory usage in MB using tracemalloc."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        current, peak = tracemalloc.get_traced_memory()
        return current / 1024 / 1024  # Convert to MB

    def test_memory_usage_during_initialization(self):
        """Test memory usage during generator initialization."""
        tracemalloc.start()
        initial_memory = self.get_memory_usage()

        with (
            patch(
                "llm_newsletter_generator.llm_newsletter_generator." "AutoTokenizer.from_pretrained"
            ),
            patch("llm_newsletter_generator.llm_newsletter_generator.pipeline"),
        ):
            generator = NewsletterGenerator(feed_url="http://example.com/rss")

        post_init_memory = self.get_memory_usage()
        memory_increase = post_init_memory - initial_memory
        tracemalloc.stop()

        # With mocked models, memory increase should be minimal
        assert (
            memory_increase < 50
        ), f"Memory increased by {memory_increase:.1f}MB during initialization"
        assert generator is not None

    @patch("feedparser.parse")
    def test_memory_usage_with_large_feeds(self, mock_parse, generator):
        """Test memory usage when processing large feeds."""
        tracemalloc.start()
        initial_memory = self.get_memory_usage()

        # Create a large feed (1000 items)
        large_entries = []
        for i in range(1000):
            large_entries.append(
                {
                    "title": f"Article {i}" + " with extra content" * 10,
                    "description": f"Long description {i} " * 50,  # ~2.5KB per description
                    "link": f"http://example.com/article/{i}",
                }
            )

        mock_parse.return_value.entries = large_entries

        # Process the large feed
        items = generator.get_items("<rss>large feed</rss>")

        post_processing_memory = self.get_memory_usage()
        memory_increase = post_processing_memory - initial_memory
        tracemalloc.stop()

        assert len(items) == 1000
        # Memory increase should be reasonable for 1000 items (~2.5MB of text data)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB for 1000 items"

    def test_memory_cleanup_after_generation(self, generator):
        """Test that memory is properly cleaned up after text generation."""
        tracemalloc.start()
        initial_memory = self.get_memory_usage()

        # Generate multiple texts
        with (
            patch("os.path.exists", return_value=False),
            patch("builtins.open"),
            patch.object(
                generator,
                "text_generation",
                return_value=[{"generated_text": "prompt + " + "generated content " * 100}],
            ),
        ):
            for i in range(10):
                result = generator.generate_text(f"Test prompt {i}")
                assert result is not None

        # Force garbage collection
        gc.collect()

        post_generation_memory = self.get_memory_usage()
        memory_increase = post_generation_memory - initial_memory
        tracemalloc.stop()

        # Memory increase should be minimal with proper cleanup
        assert (
            memory_increase < 20
        ), f"Memory increased by {memory_increase:.1f}MB after 10 generations"


class TestConcurrencyPerformance:
    """Test performance under concurrent usage scenarios."""

    @pytest.fixture
    def generator(self):
        """Returns a mocked NewsletterGenerator instance."""
        with (
            patch(
                "llm_newsletter_generator.llm_newsletter_generator.AutoTokenizer.from_pretrained"
            ),
            patch("llm_newsletter_generator.llm_newsletter_generator.pipeline"),
        ):
            yield NewsletterGenerator(feed_url="http://example.com/rss")

    def test_cache_thread_safety_simulation(self, generator):
        """Simulate concurrent cache access (single-threaded test)."""

        results = queue.Queue()
        errors = queue.Queue()

        def generate_text_worker(prompt_id):
            try:
                with (
                    patch("os.path.exists", return_value=True),
                    patch("builtins.open", mock_open(read_data="cached result")),
                ):
                    result = generator.generate_text(f"Test prompt {prompt_id}")
                    results.put((prompt_id, result))
            except Exception as e:
                errors.put((prompt_id, str(e)))

        # Simulate concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=generate_text_worker, args=(i,))
            threads.append(thread)

        start_time = time.time()
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Check results
        assert errors.empty(), f"Errors occurred: {list(errors.queue)}"
        assert results.qsize() == 5, f"Expected 5 results, got {results.qsize()}"

        # All threads should complete reasonably quickly
        assert total_time < 1.0, f"Concurrent operations took {total_time:.3f}s, expected < 1.0s"


class TestScalabilityLimits:
    """Test system behavior at scale limits."""

    @pytest.fixture
    def generator(self):
        """Returns a mocked NewsletterGenerator instance."""
        with (
            patch(
                "llm_newsletter_generator.llm_newsletter_generator.AutoTokenizer.from_pretrained"
            ),
            patch("llm_newsletter_generator.llm_newsletter_generator.pipeline"),
        ):
            yield NewsletterGenerator(feed_url="http://example.com/rss")

    @patch("feedparser.parse")
    def test_maximum_feed_size_handling(self, mock_parse, generator):
        """Test handling of very large feeds."""
        # Test with extremely large feed (5000 items)
        max_items = 5000
        large_entries = []

        for i in range(max_items):
            large_entries.append(
                {
                    "title": f"Article {i}",
                    "description": f"Description {i}",
                    "link": f"http://example.com/{i}",
                }
            )

        mock_parse.return_value.entries = large_entries

        start_time = time.time()
        items = generator.get_items("<rss>maximum size feed</rss>")
        processing_time = time.time() - start_time

        assert len(items) == max_items
        # Should handle large feeds within reasonable time
        assert processing_time < 1.0, f"Processing {max_items} items took {processing_time:.3f}s"

    def test_very_long_prompt_handling(self, generator):
        """Test handling of extremely long prompts."""
        # Create a very long prompt (10KB)
        long_prompt = "This is a very long prompt. " * 400  # ~10KB

        with (
            patch("os.path.exists", return_value=False),
            patch("builtins.open"),
            patch.object(
                generator,
                "text_generation",
                return_value=[{"generated_text": long_prompt + "Generated response"}],
            ),
        ):
            start_time = time.time()
            result = generator.generate_text(long_prompt)
            processing_time = time.time() - start_time

        assert result == "Generated response"
        # Should handle long prompts efficiently
        assert processing_time < 0.1, f"Long prompt processing took {processing_time:.3f}s"

    def test_cache_size_limits(self, generator):
        """Test cache behavior with many entries."""
        # Simulate many cache entries
        cache_entries = 100

        def mock_text_generation(prompt, **kwargs):
            return [{"generated_text": "response"}]

        with (
            patch("os.path.exists", return_value=False),
            patch("builtins.open"),
            patch.object(generator, "text_generation", side_effect=mock_text_generation),
        ):
            start_time = time.time()

            for i in range(cache_entries):
                result = generator.generate_text(f"Cache test prompt {i}")
                assert result == "response"

            total_time = time.time() - start_time

        # Should handle many cache operations efficiently
        avg_time_per_operation = total_time / cache_entries
        assert (
            avg_time_per_operation < 0.01
        ), f"Average cache operation took {avg_time_per_operation:.4f}s"
