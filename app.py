git fetch --all
git checkout 0aa345a -- app.py README.md
python -m py_compile app.py
git add app.py README.md
git commit -m "Restore clean app.py and README"
git push
