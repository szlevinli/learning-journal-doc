from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter


def get_nb_and_md_dir(src: str | Path, dst: str | Path):
    """
    获取源目录下所有的.ipynb文件并创建对应的目标目录

    参数：
        src (str或Path)：源目录的路径
        dst (str或Path)：目标目录的路径

    返回：
        生成器对象，包含文件名和目标目录路径

    需求：
        需要有转换文件的函数
    """

    src_dir_path = Path(src)
    dst_dir_path = Path(dst)

    # 验证src和dst是否是有效目录
    if not src_dir_path.is_dir() or not dst_dir_path.is_dir():
        raise ValueError("src and dst must be directories")

    # 遍历源目录下所有.ipynb文件
    for nb in src_dir_path.glob("**/*.ipynb"):
        # 获取笔记本文件所在目录的名称
        dir_name = nb.parent.stem

        # 在目标目录下创建与笔记本文件目录同名的子目录
        dst_subdir = dst_dir_path / dir_name
        dst_subdir.mkdir(parents=True, exist_ok=True)

        # 生成文件名和目标目录路径的元组
        yield nb, dst_subdir


def main():
    nb_dir = Path("../jupyter_notebooks")
    md_dir = Path("../docs")

    for nb_file, dst_dir in get_nb_and_md_dir(nb_dir, md_dir):
        nb_content = nbformat.read(nb_file, as_version=4)

        md_exporter = MarkdownExporter()

        # Ensure the directory for the embedded resources exists
        resources_dir = dst_dir / f"{nb_file.stem}_resources"
        resources_dir.mkdir(parents=True, exist_ok=True)

        res_dict = {
            "output_files_dir": str(resources_dir.stem),
        }

        (body, resources) = md_exporter.from_notebook_node(
            nb_content, resources=res_dict
        )

        # Save each resource (image) to the resources directory
        for filename, data in resources["outputs"].items():
            with open(dst_dir / filename, "wb") as f:
                f.write(data)
            # print(filename)

        with open(dst_dir / f"{nb_file.stem}.md", "w") as f:
            f.write(body)


if __name__ == "__main__":
    main()
