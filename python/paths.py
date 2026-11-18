"""
Where Calibra keeps its files (v5.0).

The program folder holds only the program. Everything Calibra writes while
running lives in one *user folder*:

    <user folder>/
        config/   calibra_session.json, calibration_config.json, instruments/,
                  setups/, local_catalogs.json, priority_targets.csv,
                  open_clusters.csv, reference_stars.csv
        cache/    catalogs/ (online query cache, local-catalogue index),
                  aavso_sequences/, apass_dr10_downloads.json,
                  time_series_cache_v2.json, dm_cache/
        output/   results (light curves, reports, period_analysis/,
                  field_variables/, logs/, plots/, nina/)
        catalogs/ APASS_DR10/ (downloads when no local catalogue folder is set)

User folder, first match wins:
    1. environment variable CALIBRA_HOME
    2. first line of the file calibra_home.txt in the program folder
       (a relative path is taken relative to the program folder, so "." gives
       the old behaviour: everything inside the program folder)
    3. Calibra in the home folder (~/Calibra)

All paths are module attributes and are read at the time of use
(paths.OUTPUT_DIR etc.), so set_user_dir() changes them for the whole program.
"""
import os
import json
import shutil

PROGRAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POINTER_FILE = os.path.join(PROGRAM_DIR, "calibra_home.txt")
DOCS_DIR = os.path.join(PROGRAM_DIR, "docs")
EXAMPLES_DIR = os.path.join(PROGRAM_DIR, "examples")
DEFAULT_USER_DIR = os.path.join(os.path.expanduser("~"), "Calibra")


def _resolve_user_dir():
    env = os.environ.get("CALIBRA_HOME", "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env)), "environment variable CALIBRA_HOME"
    try:
        with open(POINTER_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    p = os.path.expanduser(line)
                    if not os.path.isabs(p):
                        p = os.path.join(PROGRAM_DIR, p)
                    return os.path.abspath(p), "calibra_home.txt"
    except OSError:
        pass
    return DEFAULT_USER_DIR, "default"


# Defined by set_user_dir() (listed here for readers and code checkers)
USER_DIR = USER_DIR_SOURCE = CONFIG_DIR = CACHE_DIR = OUTPUT_DIR = None
SESSION_FILE = CALIBRATION_CONFIG_FILE = INSTRUMENTS_DIR = SETUPS_DIR = None
LOCAL_CATALOGS_FILE = PRIORITY_FILE = CLUSTER_FILE = REFERENCE_STARS_FILE = None
CATALOG_CACHE_DIR = AAVSO_SEQUENCES_DIR = APASS_LOG_FILE = TIME_SERIES_CACHE_FILE = DM_CACHE_DIR = None
LOGS_DIR = PLOTS_DIR = FIELD_VARIABLES_DIR = PERIOD_ANALYSIS_DIR = NINA_DIR = APASS_DR10_DIR = None


def set_user_dir(path, source="set_user_dir"):
    """(Re)define every path below the user folder `path`."""
    u = os.path.abspath(os.path.expanduser(str(path)))
    cfg, cache, out = os.path.join(u, "config"), os.path.join(u, "cache"), os.path.join(u, "output")
    globals().update(
        USER_DIR=u, USER_DIR_SOURCE=source,
        CONFIG_DIR=cfg, CACHE_DIR=cache, OUTPUT_DIR=out,
        # config
        SESSION_FILE=os.path.join(cfg, "calibra_session.json"),
        CALIBRATION_CONFIG_FILE=os.path.join(cfg, "calibration_config.json"),
        INSTRUMENTS_DIR=os.path.join(cfg, "instruments"),
        SETUPS_DIR=os.path.join(cfg, "setups"),
        LOCAL_CATALOGS_FILE=os.path.join(cfg, "local_catalogs.json"),
        PRIORITY_FILE=os.path.join(cfg, "priority_targets.csv"),
        CLUSTER_FILE=os.path.join(cfg, "open_clusters.csv"),
        REFERENCE_STARS_FILE=os.path.join(cfg, "reference_stars.csv"),
        # cache
        CATALOG_CACHE_DIR=os.path.join(cache, "catalogs"),
        AAVSO_SEQUENCES_DIR=os.path.join(cache, "aavso_sequences"),
        APASS_LOG_FILE=os.path.join(cache, "apass_dr10_downloads.json"),
        TIME_SERIES_CACHE_FILE=os.path.join(cache, "time_series_cache_v2.json"),
        DM_CACHE_DIR=os.path.join(cache, "dm_cache"),
        # output
        LOGS_DIR=os.path.join(out, "logs"),
        PLOTS_DIR=os.path.join(out, "plots"),
        FIELD_VARIABLES_DIR=os.path.join(out, "field_variables"),
        PERIOD_ANALYSIS_DIR=os.path.join(out, "period_analysis"),
        NINA_DIR=os.path.join(out, "nina"),
        # downloaded catalogue data
        APASS_DR10_DIR=os.path.join(u, "catalogs", "APASS_DR10"),
    )


set_user_dir(*_resolve_user_dir())


def output(*parts):
    """Path in the output folder (folder created)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, *parts)


def ensure_user_dirs(copy_defaults=True):
    """Create the user folder structure and copy shipped defaults that are missing."""
    for d in (CONFIG_DIR, CACHE_DIR, OUTPUT_DIR, INSTRUMENTS_DIR, SETUPS_DIR,
              CATALOG_CACHE_DIR, LOGS_DIR, PLOTS_DIR):
        os.makedirs(d, exist_ok=True)
    example = os.path.join(EXAMPLES_DIR, "reference_stars.csv")
    if copy_defaults and not os.path.exists(REFERENCE_STARS_FILE) and os.path.exists(example):
        try:
            shutil.copy2(example, REFERENCE_STARS_FILE)
        except OSError:
            pass


def write_pointer(path):
    """Store the user folder in calibra_home.txt (takes effect at the next start)."""
    with open(POINTER_FILE, "w", encoding="utf-8") as f:
        f.write("# Calibra user folder (settings, cache, results). Delete this file to use ~/Calibra.\n")
        f.write(os.path.abspath(os.path.expanduser(path)) + "\n")


# ------------------------------------------------------------ older layout (v4.x)
# v4.x wrote everything relative to the folder Calibra was started from.
_LEGACY_PREFIXES = (   # (old relative path, attribute name of the new location); longest first
    ("photometry_refstars/reference_stars.csv", "REFERENCE_STARS_FILE"),
    ("photometry_refstars/local_catalogs.json", "LOCAL_CATALOGS_FILE"),
    ("photometry_refstars/priority_targets.csv", "PRIORITY_FILE"),
    ("photometry_refstars/open_clusters.csv", "CLUSTER_FILE"),
    ("photometry_refstars/apass_dr10_downloads.json", "APASS_LOG_FILE"),
    ("photometry_refstars/aavso_sequences", "AAVSO_SEQUENCES_DIR"),
    ("photometry_refstars/APASS_DR10", "APASS_DR10_DIR"),
    ("photometry_refstars/cache", "CATALOG_CACHE_DIR"),
    ("photometry_output/time_series_cache_v2.json", "TIME_SERIES_CACHE_FILE"),
    ("photometry_output/dm_cache", "DM_CACHE_DIR"),
    ("photometry_output", "OUTPUT_DIR"),
    ("photometry_plots", "PLOTS_DIR"),
    ("calibra_session.json", "SESSION_FILE"),
    ("calibration_config.json", "CALIBRATION_CONFIG_FILE"),
    ("instruments", "INSTRUMENTS_DIR"),
    ("setups", "SETUPS_DIR"),
)


def translate_legacy(value, old_dir=None):
    """Map a path of the v4.x layout (relative, or absolute below `old_dir`) to the
    new location. Anything else is returned unchanged."""
    if not isinstance(value, str) or not value:
        return value
    v = value.replace("\\", "/")
    rel = None
    if old_dir:
        o = os.path.abspath(old_dir).replace("\\", "/").rstrip("/") + "/"
        if v.startswith(o):
            rel = v[len(o):]
    if rel is None:
        if v.startswith("./"):
            v = v[2:]
        if not os.path.isabs(value) and ":" not in v[:3]:
            rel = v
    if rel is None:
        return value
    for old, attr in _LEGACY_PREFIXES:
        if rel == old or rel.startswith(old + "/"):
            rest = rel[len(old):].lstrip("/")
            new = globals()[attr]
            return os.path.join(new, *rest.split("/")) if rest else new
    return value


def translate_legacy_tree(obj, old_dir=None):
    """translate_legacy() applied to every string in a JSON-like structure."""
    if isinstance(obj, dict):
        return {k: translate_legacy_tree(v, old_dir) for k, v in obj.items()}
    if isinstance(obj, list):
        return [translate_legacy_tree(v, old_dir) for v in obj]
    return translate_legacy(obj, old_dir)


def find_legacy_dirs():
    """Folders that contain settings or results of the v4.x layout."""
    out = []
    for d in (PROGRAM_DIR, os.getcwd()):
        d = os.path.abspath(d)
        if d in out or d == USER_DIR:
            continue
        if os.path.isfile(os.path.join(d, "calibra_session.json")) or \
           os.path.isdir(os.path.join(d, "photometry_output")):
            out.append(d)
    return out


def migrate_from(old_dir, overwrite=False):
    """Copy settings, caches and results of a v4.x Calibra folder into the user folder.

    Nothing is deleted or moved; existing files in the user folder are kept unless
    overwrite=True. Paths inside JSON settings files are rewritten to the new layout.
    Returns (number of files copied, number skipped, list of messages)."""
    old_dir = os.path.abspath(old_dir)
    ensure_user_dirs(copy_defaults=False)
    copied, skipped, msgs = 0, 0, []
    for root, dirs, files in os.walk(old_dir):
        rel_root = os.path.relpath(root, old_dir).replace("\\", "/")
        top = rel_root.split("/")[0] if rel_root != "." else ""
        if rel_root == ".":
            dirs[:] = [d for d in dirs if d in ("photometry_output", "photometry_plots",
                                               "photometry_refstars", "instruments", "setups")]
            files = [f for f in files if f in ("calibra_session.json", "calibration_config.json")]
        elif top not in ("photometry_output", "photometry_plots", "photometry_refstars",
                         "instruments", "setups"):
            continue
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")
        for fn in files:
            rel = fn if rel_root == "." else rel_root + "/" + fn
            dest = translate_legacy(rel)
            if dest == rel:      # not part of the old layout
                continue
            src = os.path.join(root, fn)
            if os.path.exists(dest) and not overwrite and not _is_shipped_default(dest):
                skipped += 1
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                if fn.lower().endswith(".json"):
                    try:
                        with open(src, encoding="utf-8") as f:
                            data = json.load(f)
                        data = translate_legacy_tree(data, old_dir)
                        with open(dest, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=4)
                    except (ValueError, UnicodeDecodeError):
                        shutil.copy2(src, dest)
                else:
                    shutil.copy2(src, dest)
                copied += 1
            except OSError as e:
                msgs.append(f"could not copy {rel}: {e}")
    ensure_user_dirs()
    msgs.insert(0, f"{copied} file(s) copied from {old_dir} to {USER_DIR}"
                   + (f", {skipped} already present and kept" if skipped else ""))
    return copied, skipped, msgs


def _is_shipped_default(dest):
    """True for an unchanged copy of a file shipped with Calibra (may be replaced)."""
    if dest != REFERENCE_STARS_FILE:
        return False
    try:
        with open(dest, "rb") as a, open(os.path.join(EXAMPLES_DIR, "reference_stars.csv"), "rb") as b:
            return a.read() == b.read()
    except OSError:
        return False


def describe():
    return (f"User folder: {USER_DIR}  ({USER_DIR_SOURCE})\n"
            f"  settings: {CONFIG_DIR}\n  cache:    {CACHE_DIR}\n  results:  {OUTPUT_DIR}")
