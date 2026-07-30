from typing import TypedDict


class CoreOverride(TypedDict, total=False):
    npin: int
    nax: int


type FileOverrides = dict[str, CoreOverride]
