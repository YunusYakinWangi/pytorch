#!/bin/bash

# This is where the local pytorch install in the docker image is located
pt_checkout="${GITHUB_WORKSPACE:-/var/lib/jenkins/workspace}"

source "$pt_checkout/.ci/pytorch/common_utils.sh"

echo "python_doc_push_script.sh: Invoked with $*"

set -ex -o pipefail

# for statements like ${1:-${DOCS_INSTALL_PATH:-docs/}}
# the order of operations goes:
#   1. Check if there's an argument $1
#   2. If no argument check for environment var DOCS_INSTALL_PATH
#   3. If no environment var fall back to default 'docs/'

# NOTE: It might seem weird to gather the second argument before gathering the first argument
#       but since DOCS_INSTALL_PATH can be derived from DOCS_VERSION it's probably better to
#       try and gather it first, just so we don't potentially break people who rely on this script
# Argument 2: What version of the docs we are building.
version="${2:-${DOCS_VERSION:-main}}"
if [ -z "$version" ]; then
echo "error: python_doc_push_script.sh: version (arg2) not specified"
  exit 1
fi

# Argument 1: Where to copy the built documentation to
# (pytorch_docs/$install_path)
install_path="${1:-${DOCS_INSTALL_PATH:-${DOCS_VERSION}}}"
if [ -z "$install_path" ]; then
echo "error: python_doc_push_script.sh: install_path (arg1) not specified"
  exit 1
fi

is_main_doc=false
if [ "$version" == "main" ]; then
  is_main_doc=true
fi

# Argument 3: The branch to push to. Usually is "site"
branch="${3:-${DOCS_BRANCH:-site}}"
if [ -z "$branch" ]; then
echo "error: python_doc_push_script.sh: branch (arg3) not specified"
  exit 1
fi

echo "install_path: $install_path  version: $version"


build_docs () {
  set +e
  # Don't pipe through tee: sphinx -j auto forks workers that inherit
  # the pipe fd and hold it open after sphinx exits, causing tee to
  # block forever. Write to a file and tail with --pid so it exits
  # (after draining) when make finishes.
  make "$1" > /tmp/docs_build.txt 2>&1 &
  local make_pid=$!
  tail -f --pid=$make_pid /tmp/docs_build.txt
  wait $make_pid
  code=$?
  if [ $code -ne 0 ]; then
    set +x
    echo =========================
    grep "WARNING:" /tmp/docs_build.txt
    echo =========================
    echo Docs build failed. If the failure is not clear, scan back in the log
    echo for any WARNINGS or for the line "build finished with problems"
    echo "(tried to echo the WARNINGS above the ==== line)"
    echo =========================
  fi
  set -ex -o pipefail
  return $code
}


git clone https://github.com/pytorch/docs pytorch_docs -b "$branch" --depth 1
pushd pytorch_docs

export LC_ALL=C
export PATH=/opt/conda/bin:$PATH
if [ -n "$ANACONDA_PYTHON_VERSION" ]; then
  export PATH=/opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/bin:$PATH
fi

rm -rf pytorch || true

# Get all the documentation sources, put them in one place
pushd "$pt_checkout"
pushd docs

# Build the docs
if [ "$is_main_doc" = true ]; then
  build_docs html || exit $?

  # Run coverage check without parallel workers since it's a quick
  # check that doesn't need parallelism, and avoids re-triggering the
  # expensive parallel read/write machinery.
  SPHINXOPTS="-WT --keep-going" make coverage
  # Now we have the coverage report, we need to make sure it is empty.
  # Sphinx 7.2.6+ format: python.txt contains a statistics table with a TOTAL row
  # showing the undocumented count in the third column.
  # Example: | TOTAL | 99.83% | 2 |
  #
  # Also: see docs/source/conf.py for "coverage_ignore*" items, which should
  # be documented then removed from there.

  # Extract undocumented count from TOTAL row in Sphinx 7.2.6 statistics table
  # The table format is: | Module | Coverage | Undocumented |
  # Extract the third column (undocumented count) from the TOTAL row
  undocumented=$(grep "| TOTAL" build/coverage/python.txt | awk -F'|' '{print $4}' | tr -d ' ')

  if [ -z "$undocumented" ] || ! [[ "$undocumented" =~ ^[0-9]+$ ]]; then
    echo coverage output not found
    exit 1
  elif [ "$undocumented" -gt 0 ]; then
    set +x  # Disable command echoing for cleaner output
    echo ""
    echo "====================="
    echo "UNDOCUMENTED OBJECTS:"
    echo "====================="
    echo ""
    # Find the line number of the TOTAL row and print only what comes after it
    total_line=$(grep -n "| TOTAL" build/coverage/python.txt | cut -d: -f1)
    if [ -n "$total_line" ]; then
      # Print only the detailed list (skip the statistics table)
      tail -n +$((total_line + 2)) build/coverage/python.txt
    else
      # Fallback to showing entire file if TOTAL line not found
      cat build/coverage/python.txt
    fi
    echo ""
    echo "Make sure you've updated relevant .rsts in docs/source!"
    echo "You can reproduce locally by running 'cd docs && make coverage && tail -n +\$((grep -n \"| TOTAL\" build/coverage/python.txt | cut -d: -f1) + 2)) build/coverage/python.txt'"
    set -x  # Re-enable command echoing
    exit 1
  fi
else
  # skip coverage, format for stable or tags
  build_docs html-stable || exit $?
fi

# Generate llms-full.txt from built HTML (nightly/release only).
# This is done as a separate step because converting ~2800 HTML pages
# to markdown is too slow to run inside the Sphinx build itself.
if [[ "${WITH_PUSH:-}" == true ]]; then
  echo "Generating llms-full.txt from built HTML pages..."
  python -c "
import sys
sys.path.insert(0, '$(python -c "import pytorch_sphinx_theme2; import os; print(os.path.dirname(pytorch_sphinx_theme2.__file__))")')
from llm_generation import _html_to_markdown
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

outdir = Path('build/html')
html_files = sorted(outdir.rglob('*.html'))
total = len(html_files)
print(f'Converting {total} HTML files to markdown...')

def convert(html_path):
    try:
        content = html_path.read_text(encoding='utf-8')
        md = _html_to_markdown(content)
        if md.strip():
            md_path = html_path.with_suffix('.md')
            md_path.write_text(md, encoding='utf-8')
            return (html_path.stem, md)
    except Exception as e:
        print(f'Warning: {html_path}: {e}')
    return (html_path.stem, None)

results = {}
max_workers = min(os.cpu_count() or 4, 8)
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(convert, p): p for p in html_files}
    done = 0
    interval = max(1, total // 10)
    for f in as_completed(futures):
        name, md = f.result()
        if md:
            results[name] = md
        done += 1
        if done % interval == 0 or done == total:
            print(f'  {done}/{total} files processed')

sections = ['# PyTorch\n\n> PyTorch documentation.\n']
for name, md in sorted(results.items()):
    sections.append(f'\n---\n\n{md}')
full_path = outdir / 'llms-full.txt'
full_path.write_text('\n'.join(sections), encoding='utf-8')
print(f'Generated llms-full.txt ({len(results)} pages) at: {full_path}')
"
fi

# Move them into the docs repo
popd
popd
git rm -rf "$install_path" || true
mv "$pt_checkout/docs/build/html" "$install_path"

git add "$install_path" || true
git status
git config user.email "soumith+bot@pytorch.org"
git config user.name "pytorchbot"
# If there aren't changes, don't make a commit; push is no-op
git commit -m "Generate Python docs from pytorch/pytorch@${GITHUB_SHA}" || true
git status

if [[ "${WITH_PUSH:-}" == true ]]; then
  # push to a temp branch first to trigger CLA check and satisfy branch protections
  git push -u origin HEAD:pytorchbot/temp-branch-py -f
  git push -u origin HEAD^:pytorchbot/base -f
  sleep 30
  git push -u origin "${branch}"
fi

popd
