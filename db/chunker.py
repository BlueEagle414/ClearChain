def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text_to_split: str, current_separators: list[str]) -> list[str]:
        if len(text_to_split) <= chunk_size:
            return [text_to_split]

        # Find the highest priority separator that actually exists in the text
        separator = current_separators[-1]
        next_separators = []

        for i, sep in enumerate(current_separators):
            if sep == "":
                separator = sep
                break
            if sep in text_to_split:
                separator = sep
                next_separators = current_separators[i + 1:]
                break

        if separator == "":
            splits = list(text_to_split)
        else:
            splits = text_to_split.split(separator)

        chunks = []
        current_chunk = []
        current_len = 0

        for split in splits:
            if len(split) > chunk_size:
                # If we have accumulated a chunk, save it before processing the oversized split
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_len = 0

                chunks.extend(_split(split, next_separators))
                continue

            added_len = len(split) + (len(separator) if current_chunk else 0)

            if current_len + added_len > chunk_size and current_chunk:
                chunks.append(separator.join(current_chunk))

                while current_chunk and current_len > overlap:
                    removed = current_chunk.pop(0)
                    current_len -= len(removed) + (len(separator) if current_chunk else 0)

            current_chunk.append(split)
            current_len += len(split) + (len(separator) if len(current_chunk) > 1 else 0)

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    return _split(text, separators)
