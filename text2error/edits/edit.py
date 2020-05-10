from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import


class _TextEdit(NamedTuple):
    text: str
    start: int
    end: int


class TextEditBackwardEditException(ValueError):
    ...


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
        text_len, edits_len = len(text), len(edits)
        if edits_len == 0:
            return text

        builder = []
        i, j, o = 0, 0, 0
        while i < text_len:
            if j == edits_len:
                builder.append(text[i:])
                break
            if edits[j].start < i + o:
                raise TextEditBackwardEditException(
                    "Illegal backward edit found at index %d: "
                    "Applying %s when the iterator is at %d"
                    % (j, edits[j], text_len + o)
                )
            if edits[j].end > text_len + o:
                builder.append(text[i:])
                raise TextEditOutOfBoundsException(
                    "Illegal out-of-bounds edit found at index %d: "
                    "Applying %s to `%s`" % (j, edits[j], "".join(builder))
                )
            if i + o == edits[j].start:
                if edits[j].text != "":
                    builder.append(edits[j].text)
                do = len(edits[j].text) - (edits[j].end - edits[j].start)
                i = edits[j].end - o
                j += 1
                o += do
                continue
            builder.append(text[i : edits[j].start - o])
            i = edits[j].start - o
        while j < edits_len:
            if edits[j].start < text_len + o:
                raise TextEditBackwardEditException(
                    "Illegal backward edit found at index %d: "
                    "Applying %s when the iterator is at %d"
                    % (j, edits[j], text_len + o)
                )
            if edits[j].end > text_len + o:
                raise TextEditOutOfBoundsException(
                    "Illegal out-of-bounds edit found at index %d: "
                    "Applying %s to `%s`" % (j, edits[j], "".join(builder))
                )
            builder.append(edits[j].text)
            o += len(edits[j].text)
            j += 1

        return "".join(builder)
