"""Backward-compatible entrypoint.

If you run this as a script:
  python summary_based_classifier/run.py ...
Python sets sys.path[0] to the script directory (the package dir), so importing
`summary_based_classifier.*` will fail unless the project root is also on sys.path.

Preferred:
  python -m summary_based_classifier.run ...
  python -m summary_based_classifier.cli.run_oracle_sft ...
"""
import sys
import os


def main():

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(pkg_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from summary_based_classifier.cli.run_oracle_pipeline import main as oracle_main
    return oracle_main()


if __name__ == "__main__":
    main()
