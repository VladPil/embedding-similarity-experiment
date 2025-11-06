#!/usr/bin/env python3

import re
import asyncio
from src.text_domain.services.chunking_service import ChunkingService
from src.text_domain.entities.chunking_strategy import ChunkingStrategy

async def debug_chunking():
    service = ChunkingService()
    strategy = ChunkingStrategy.create_sentence_based(
        strategy_id="test-sentence",
        sentences_per_chunk=2,
        overlap_sentences=0
    )

    text = "Первое предложение. Второе предложение. Третье предложение. Четвертое предложение."

    print(f"Text: {text}")
    print(f"Text length: {len(text)}")
    print(f"Strategy: {strategy}")
    print(f"Strategy use_sentence_boundaries: {strategy.use_sentence_boundaries}")
    print(f"Strategy use_paragraph_boundaries: {strategy.use_paragraph_boundaries}")
    print(f"Strategy sentences_per_chunk: {getattr(strategy, 'sentences_per_chunk', None)}")
    print(f"Strategy balance_chunks: {strategy.balance_chunks}")
    print(f"Strategy overlap_percentage: {strategy.overlap_percentage}")

    # Проверим регулярное выражение
    SENTENCE_ENDINGS = re.compile(r'[.!?]+[\s\n]+')
    sentence_positions = [0]
    for match in SENTENCE_ENDINGS.finditer(text):
        sentence_positions.append(match.end())
    if sentence_positions[-1] != len(text):
        sentence_positions.append(len(text))

    print(f"Sentence positions: {sentence_positions}")

    # Проверим sentences
    for i in range(len(sentence_positions) - 1):
        start = sentence_positions[i]
        end = sentence_positions[i + 1]
        sentence = text[start:end]
        print(f"Sentence {i}: '{sentence}' (positions {start}-{end})")

    # Ручная проверка алгоритма
    sentences_per_chunk = 2
    manual_chunks = []
    sentence_idx = 0

    print(f"\nManual algorithm simulation:")
    while sentence_idx < len(sentence_positions) - 1:
        end_sentence_idx = min(sentence_idx + sentences_per_chunk, len(sentence_positions) - 1)

        chunk_start = sentence_positions[sentence_idx]
        chunk_end = sentence_positions[end_sentence_idx]

        print(f"Iteration: sentence_idx={sentence_idx}, end_sentence_idx={end_sentence_idx}")
        print(f"  chunk_start={chunk_start}, chunk_end={chunk_end}")
        print(f"  chunk_text='{text[chunk_start:chunk_end]}'")

        manual_chunks.append((chunk_start, chunk_end, text[chunk_start:chunk_end]))
        sentence_idx = end_sentence_idx

    print(f"\nManual chunks: {len(manual_chunks)}")
    for i, (start, end, content) in enumerate(manual_chunks):
        print(f"Manual chunk {i}: '{content}' (positions {start}-{end})")

    chunks = await service.chunk_text(text, strategy)

    print(f"\nActual chunks count: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Actual chunk {i}: '{chunk.content}' (positions {chunk.start_pos}-{chunk.end_pos})")

if __name__ == "__main__":
    asyncio.run(debug_chunking())