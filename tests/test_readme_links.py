"""The README is also the PyPI description, where relative links break: every link must be absolute."""
import os
import re

README = os.path.join(os.path.dirname(__file__), "..", "README.md")


def test_readme_links_are_absolute():
    links = re.findall(r"\]\(([^)]+)\)", open(README, encoding="utf-8").read())
    bad = [u for u in links if not u.startswith(("https://", "http://", "#", "mailto:"))]
    assert not bad, f"relative links in README.md (broken on PyPI): {bad}"
