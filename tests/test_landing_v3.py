"""The interactive landing for investors (landing/, served at /landing): every scene playable, the page self-contained.

It loads nothing from other sites, because the server sends a policy that only allows its own files, and because at a
stand the network may be gone."""
import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANDING = ROOT / "landing"
SITE = ROOT / "remote/deploy/vps/infinito-remote.caddy"


def config():
    text = (LANDING / "escenas.js").read_text(encoding="utf-8")
    assert "window.INFINITO_LANDING = {" in text
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.scripts_with_code = 0
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        self._in_script = tag == "script"

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False

    def handle_data(self, data):
        if self._in_script and data.strip():
            self.scripts_with_code += 1


def page():
    parser = Page()
    parser.feed((LANDING / "index.html").read_text(encoding="utf-8"))
    return parser


def test_the_scenes_of_the_brief_are_all_there_between_four_and_six():
    scenes = config()["scenes"]
    ids = [scene["id"] for scene in scenes]
    assert 4 <= len(scenes) <= 6 and len(set(ids)) == len(ids)
    assert {"identidad", "foto", "ordenador", "jardin", "robot"} <= set(ids)


def test_every_flow_points_at_something_the_page_can_draw():
    cfg = config()
    ids = {scene["id"] for scene in cfg["scenes"]}
    arts = set(re.findall(r"^\s{4}(\w+):\s*$", (LANDING / "landing.js").read_text(encoding="utf-8"), re.M))
    assert {step["id"] for step in cfg["steps"]} == {"recoge", "recuerda", "analiza", "actua"}
    assert cfg["idle"]["title"].strip() and cfg["idle"]["text"].strip() and cfg["apps"]
    for scene in cfg["scenes"]:
        assert scene["side"] in {"left", "right"} and scene["art"] in arts, scene["id"]
        assert scene["packet"].strip() and scene["action"].strip() and scene["explain"].strip(), scene["id"]
        assert scene.get("memory") or scene.get("recalls") in ids, scene["id"]
        if scene.get("packetArt"):
            assert scene["packetArt"] in arts
        output = scene["output"]
        if output:
            assert output["to"] in {"apps", "self"} | ids and output["result"].strip(), scene["id"]
            assert output["to"] == "apps" or output["packet"].strip(), scene["id"]


def test_the_page_loads_only_its_own_files_and_runs_under_a_strict_policy():
    parsed = page()
    for tag, attrs in parsed.tags:
        for name in ("src", "href"):
            value = attrs.get(name)
            if value is None or tag == "a" or value.startswith("#"):
                continue
            assert "://" not in value and not value.startswith("//"), (tag, value)
            assert (LANDING / value).is_file(), value
        assert "style" not in attrs, tag  # no inline styles: the policy is style-src 'self'
        assert not any(name.startswith("on") for name in attrs), tag  # no inline handlers
    assert parsed.scripts_with_code == 0  # no inline scripts: the policy is script-src 'self'
    for name in ("landing.css", "landing.js", "escenas.js"):
        text = (LANDING / name).read_text(encoding="utf-8")
        assert not re.search(r"""url\(\s*['"]?(https?:)?//""", text), name
        assert "@import" not in text and "fetch(" not in text and "XMLHttpRequest" not in text, name


def test_every_anchor_in_the_page_has_its_section():
    ids = {attrs["id"] for _, attrs in page().tags if "id" in attrs}
    for tag, attrs in page().tags:
        href = attrs.get("href", "")
        if tag == "a" and href.startswith("#"):
            assert href[1:] in ids, href


def test_links_out_of_the_page_go_only_to_infinito_and_its_code():
    allowed = ("https://infinitomemory.com/", "https://github.com/webtilians/principiodelTodo")
    for tag, attrs in page().tags:
        href = attrs.get("href", "")
        if tag == "a" and "://" in href:
            assert href.startswith(allowed), href


def test_the_site_serves_the_landing_folder_at_slash_landing_with_its_policy():
    site = SITE.read_text(encoding="utf-8")
    assert "redir /landing /landing/ 308" in site
    block = site[site.index("handle_path /landing/*"):]
    block = block[:block.index("\n\t}\n")]
    assert "root * /opt/infinito-remote/landing" in block and "file_server" in block
    assert "Content-Security-Policy" in block and "default-src 'none'" in block and "script-src 'self'" in block
    # It comes before the catch-all that hands everything else to the memory server.
    assert site.index("handle_path /landing/*") < site.index("\thandle {\n")
