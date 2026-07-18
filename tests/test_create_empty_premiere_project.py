import os

import pytest

from premiere_ai import create_empty_premiere_project as cepp


@pytest.fixture
def fake_template(tmp_path, monkeypatch):
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    (template_dir / "Untitled.prproj").write_text("project data")
    (template_dir / "Adobe Premiere Pro Auto-Save").mkdir()
    monkeypatch.setattr(cepp, "TEMPLATE_DIR", str(template_dir))
    return template_dir


def test_create_project_copies_and_renames(fake_template, tmp_path):
    base_dir = tmp_path / "base"
    dest = cepp.create_project("vlog0002", base_dir=str(base_dir))

    assert dest == os.path.join(str(base_dir), "vlog0002")
    assert os.path.isfile(os.path.join(dest, "vlog0002.prproj"))
    assert not os.path.exists(os.path.join(dest, "Untitled.prproj"))
    assert os.path.isdir(os.path.join(dest, "Adobe Premiere Pro Auto-Save"))


def test_create_project_nests_under_series(fake_template, tmp_path):
    base_dir = tmp_path / "base"
    dest = cepp.create_project("episode 2", series_name="vlog", base_dir=str(base_dir))

    assert dest == os.path.join(str(base_dir), "vlog", "episode 2")
    assert os.path.isfile(os.path.join(dest, "episode 2.prproj"))


def test_create_project_refuses_to_overwrite_existing(fake_template, tmp_path):
    base_dir = tmp_path / "base"
    cepp.create_project("vlog0002", base_dir=str(base_dir))

    with pytest.raises(RuntimeError, match="already exists"):
        cepp.create_project("vlog0002", base_dir=str(base_dir))


def test_create_project_missing_template_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(cepp, "TEMPLATE_DIR", str(tmp_path / "does-not-exist"))

    with pytest.raises(RuntimeError, match="Template directory not found"):
        cepp.create_project("vlog0002", base_dir=str(tmp_path / "base"))
