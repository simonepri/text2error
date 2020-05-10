from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import


class _TextEdit(NamedTuple):
    text: str
    start: int
    end: int


class TextEditOutOfBoundsException(ValueError):
    ...


class TextEdit(_TextEdit):
    def __new__(
        cls, text: str, start: Optional[int] = None, end: Optional[int] = None
    ) -> "TextEdit":
        if start is None and end is None:
            raise ValueError("One between start and end must be provided")
        if start is None:
            start = end
        if end is None:
            end = start
        start = cast(int, start)
        end = cast(int, end)

        if start < 0 or end < 0:
            raise ValueError("The start and end position must be non negative")
        if start > end:
            raise ValueError("The start position can't be after the end")
        return super().__new__(cls, text, start, end)

    @staticmethod
    def apply(text: str, edits: List["TextEdit"]) -> str:
        # pylint: disable=invalid-name
        edits_len = len(edits)
        j = 0
        while j < edits_len:
            i = 0
            o = 0
            builder = []
            text_len = len(text)
            while j < edits_len and i < text_len:
                if edits[j].end > text_len + o:
                    break  # Out-of-bounds.
                if edits[j].start < i + o:
                    break  # Backward edit.
                if edits[j].start > i + o:
                    builder.append(text[i : edits[j].start - o])
                if edits[j].text != "":
                    builder.append(edits[j].text)
                i = edits[j].end - o
                o += len(edits[j].text) - (edits[j].end - edits[j].start)
                j += 1
            if i < text_len:
                builder.append(text[i:])
            if j < edits_len and edits[j].end > text_len + o:
                builder.append(text[i:])
                raise TextEditOutOfBoundsException(
                    "Illegal out-of-bounds edit found at index %d: "
                    "Applying %s to `%s`" % (j, edits[j], "".join(builder))
                )
            while j < edits_len and edits[j].start == edits[j].end == text_len + o:
                # Insertion at the end.
                builder.append(edits[j].text)
                o += len(edits[j].text)
                j += 1
            text = "".join(builder)

        return text
