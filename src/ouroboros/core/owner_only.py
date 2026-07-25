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
import errno
import os
from pathlib import Path
import stat
from uuid import uuid4

#: Files: readable and writable by the owner only.
OWNER_ONLY_FILE = 0o600
#: Directories: additionally traversable by the owner only.
OWNER_ONLY_DIR = 0o700


def fsync_parent_directory(file_path: Path) -> bool:
    """Flush the directory entry so the rename itself survives a crash.

    Public because durability has to be RETRYABLE (round-66): a record whose
    content reached disk but whose directory entry was not confirmed is
    already readable and already terminal, so the recovery that helps is
    flushing again — not rewriting content that is not the problem.

    Returns whether durability could be confirmed. A filesystem that cannot
    fsync a directory (``EINVAL``/``ENOTSUP``) is reported as confirmed: it
    never owed the guarantee, so treating it as a failure would make every
    write on that filesystem look suspect.
    """
    if os.name != "posix":
        return True

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        directory_fd = os.open(file_path.parent, flags)
    except OSError as error:
        return error.errno in (errno.EINVAL, errno.ENOTSUP)
    durability_confirmed = True
    try:
        try:
            os.fsync(directory_fd)
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOTSUP):
                durability_confirmed = False
    finally:
        try:
            os.close(directory_fd)
        except OSError:
            durability_confirmed = False
    return durability_confirmed


def write_owner_only(path: Path, text: str, *, encoding: str = "utf-8") -> bool:
    """Write ``text`` to ``path`` as a durable owner-only file.

    The content is written to a NEW file created at ``0600`` and then renamed
    over the target, so the mode is established at creation and never depends
    on repairing an existing file's permissions. That matters because the
    repair can fail — a filesystem without chmod, an EPERM on a file owned by
    someone else — and the earlier version suppressed that failure and wrote
    anyway, leaving the new secret at the old ``0644`` (round-61). Here a
    failure to establish the mode is a failure to write: nothing sensitive
    reaches disk at a wider mode, ever.

    The replacement is atomic, so a reader never observes a partial file, and
    the temporary lives beside the target so the rename stays within one
    filesystem. Contents and directory entry are both fsync'd, and the return
    value reports whether that durability could be confirmed — callers that
    persist state a user would notice losing should log when it could not.

    Establishing the mode at creation is also what keeps an *existing*
    world-readable file from staying that way. Preserving the previous mode,
    the usual behaviour for an atomic-write helper, would carry a `0644` file
    written by an older version forward forever.

    Directories are not touched: a caller may write into a directory that is
    not this package's to re-permission. Call :func:`secure_directory` only
    for directories this package creates and owns.
    """
    target = Path(path)
    tmp_path = target.with_name(f".{target.name}.tmp-{uuid4().hex}")
    # Held only until a file object takes ownership of it. If wrapping the
    # descriptor fails, nothing else will ever close it, so the cleanup path
    # has to.
    raw_fd: int | None = None
    try:
        raw_fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, OWNER_ONLY_FILE)
        handle = os.fdopen(raw_fd, "w", encoding=encoding)
        raw_fd = None
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if stat.S_IMODE(os.stat(tmp_path).st_mode) != OWNER_ONLY_FILE:
            # A filesystem that cannot represent the mode must not receive the
            # content at all.
            raise OSError(f"cannot create {target} with owner-only permissions on this filesystem")
        os.replace(tmp_path, target)
    except BaseException:
        if raw_fd is not None:
            with suppress(OSError):
                os.close(raw_fd)
        with suppress(OSError):
            os.unlink(tmp_path)
        raise
    return fsync_parent_directory(target)


def secure_directory(path: Path) -> None:
    """Create ``path`` if needed and make it owner-only.

    Call this ONLY for directories this package creates and owns — the
    interview state directory and the fan-out registry directory. A Seed is
    written wherever the caller asks, which may be a shared project
    directory, and narrowing that from 0755 to 0700 would be this package
    changing something that is not its own (round-60). The Seed file itself is
    still owner-only through :func:`write_owner_only`.

    ``mkdir``'s mode argument is ignored when the directory already exists, so
    an inherited `0755` state directory keeps its permissions unless it is
    chmod'd explicitly. Failure is suppressed: a directory we do not own is
    not ours to re-permission, and refusing to run there would be worse than
    proceeding with the file mode we do control.
    """
    path.mkdir(parents=True, exist_ok=True, mode=OWNER_ONLY_DIR)
    with suppress(OSError):
        os.chmod(path, OWNER_ONLY_DIR)
