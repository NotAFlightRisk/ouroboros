"""Owner-only writes for files that hold interview and data content.

Interview transcripts, Seeds, and fan-out records all carry whatever a data
lookup returned and whatever the user confirmed about it. They are written
under the process umask by default, which on a typical `022` leaves them
world-readable for their whole lifetime — indefinitely, in the case of
interview state and Seeds.

The protection cannot be an instruction to the writer to be careful: every
call site would have to remember, and one that forgets is invisible. It is a
function, applied at every site that persists this class of content, that
creates the file with mode `0600` rather than fixing the mode afterwards — so
the content never exists at the umask default even briefly.
"""

from __future__ import annotations

from contextlib import suppress
import os
from pathlib import Path

#: Files: readable and writable by the owner only.
OWNER_ONLY_FILE = 0o600
#: Directories: additionally traversable by the owner only.
OWNER_ONLY_DIR = 0o700


def write_owner_only(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` as an owner-only file.

    The mode is applied at CREATION, not after the write, so the content is
    never briefly present at the umask default. Directories are not created
    here — call :func:`secure_directory` for the parent when it is this
    package's to own.
    """
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, OWNER_ONLY_FILE)
    with os.fdopen(descriptor, "w", encoding=encoding) as handle:
        handle.write(text)


def secure_directory(path: Path) -> None:
    """Create ``path`` if needed and make it owner-only.

    ``mkdir``'s mode argument is ignored when the directory already exists, so
    an inherited `0755` state directory keeps its permissions unless it is
    chmod'd explicitly. Failure is suppressed: a directory we do not own is
    not ours to re-permission, and refusing to run there would be worse than
    proceeding with the file mode we do control.
    """
    path.mkdir(parents=True, exist_ok=True, mode=OWNER_ONLY_DIR)
    with suppress(OSError):
        os.chmod(path, OWNER_ONLY_DIR)
