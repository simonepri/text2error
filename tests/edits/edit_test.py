# pylint: disable=missing-module-docstring,missing-function-docstring,unused-variable,too-many-locals,too-many-statements
import pytest  # pylint: disable=unused-import

from text2error.edits.edit import TextEdit
from text2error.edits.edit import (
    TextEditBackwardEditException,
    TextEditOutOfBoundsException,
)


def describe_init():
    def should_throw_an_exception_when_both_start_and_end_are_none():
        with pytest.raises(ValueError):
            TextEdit("")

    def should_not_throw_an_exception_when_end_is_none():
        edit = TextEdit("", start=0)
        assert edit.text == ""
        assert edit.start == 0
        assert edit.end == 0

    def should_not_throw_an_exception_when_start_is_none():
        edit = TextEdit("", end=0)
        assert edit.text == ""
        assert edit.start == 0
        assert edit.end == 0

    def should_throw_an_exception_when_start_or_end_are_negative():
        with pytest.raises(ValueError):
            edit = TextEdit("", start=-1, end=0)
        with pytest.raises(ValueError):
            edit = TextEdit("", start=0, end=-1)

    def should_throw_an_exception_when_start_is_after_end():
        with pytest.raises(ValueError):
            edit = TextEdit("", start=1, end=0)


def describe_apply():
    def should_throw_an_exception_for_invalid_edits():
        text = "This is an example text."
        edit = TextEdit("", start=0, end=len(text) + 1)
        with pytest.raises(TextEditOutOfBoundsException):
            TextEdit.apply(text, [edit])
        edit = TextEdit(" World!", start=6, end=6)
        with pytest.raises(TextEditOutOfBoundsException):
            TextEdit.apply("Hello", [edit])
        edit = TextEdit("", start=0, end=1)
        with pytest.raises(TextEditOutOfBoundsException):
            TextEdit.apply("A", [edit, edit])
        edit_1 = TextEdit("", start=1, end=2)
        edit_2 = TextEdit("", start=0, end=1)
        with pytest.raises(TextEditBackwardEditException):
            TextEdit.apply("AB", [edit_1, edit_2])
        with pytest.raises(TextEditBackwardEditException):
            TextEdit.apply("ABC", [edit_1, edit_2, edit_1])

    def should_work_for_a_single_edit():
        # pos:  012345678901234567890123
        text = "This is an example text."

        edit = TextEdit("", start=0, end=5)
        assert TextEdit.apply(text, [edit]) == "is an example text."
        edit = TextEdit("not ", start=8, end=8)
        assert TextEdit.apply(text, [edit]) == "This is not an example text."
        edit = TextEdit("..", start=len(text), end=len(text))
        assert TextEdit.apply(text, [edit]) == "This is an example text..."

        edit = TextEdit("", start=0, end=0)
        assert TextEdit.apply(text, [edit]) == text
        edit = TextEdit("", start=len(text), end=len(text))
        assert TextEdit.apply(text, [edit]) == text
        edit = TextEdit("", start=0, end=len(text))
        assert TextEdit.apply(text, [edit]) == ""

        edit = TextEdit("", start=0, end=0)
        assert TextEdit.apply("", [edit]) == ""
        edit = TextEdit("A", start=0, end=0)
        assert TextEdit.apply("", [edit]) == "A"
