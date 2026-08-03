from pathlib import Path

from scripts.build_kinofail_realistic_c2_v5_features import index_t3_episode_dirs


def test_index_t3_episode_dirs_follows_episode_leaf_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source" / "episode_a"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text("{}\n")
    operator = tmp_path / "union" / "scene" / "c2_t3" / "material" / "operator"
    operator.mkdir(parents=True)
    (operator / "episode_a").symlink_to(source, target_is_directory=True)

    indexed = index_t3_episode_dirs(tmp_path / "union")

    assert indexed == {"episode_a": operator / "episode_a"}
