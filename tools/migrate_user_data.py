"""
Copy settings, caches and results of an earlier Calibra version (v4.x, everything in
the program folder) into the Calibra 5 user folder.

    python tools/migrate_user_data.py <old Calibra folder> [--overwrite]

Copied: calibra_session.json, calibration_config.json, instruments/, setups/,
photometry_refstars/ (local-catalogue settings, cache, AAVSO sequences, lists),
photometry_output/ and photometry_plots/. Paths inside the JSON settings files are
changed to the new locations. Nothing is deleted; files that already exist in the
user folder are kept unless --overwrite is given.

The user folder is ~/Calibra, or the folder in calibra_home.txt in the program
folder, or the environment variable CALIBRA_HOME (see python/paths.py).
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from python import paths  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('old_dir', help='folder of the earlier Calibra version')
    ap.add_argument('--overwrite', action='store_true', help='replace files that already exist in the user folder')
    a = ap.parse_args(argv)
    if not os.path.isdir(a.old_dir):
        print(f"Not a folder: {a.old_dir}")
        return 1
    print(paths.describe())
    _, _, msgs = paths.migrate_from(a.old_dir, overwrite=a.overwrite)
    print("\n".join(msgs))
    return 0


if __name__ == '__main__':
    sys.exit(main())
