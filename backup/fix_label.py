import os
import xml.etree.ElementTree as ET

def correct_xml_files(directory_path):
    """
    修正XML文件中的标签文本，删除未修改的XML文件
    """
    modified_files = 0
    total_replacements = 0

    if not os.path.exists(directory_path):
        print(f"目录 {directory_path} 不存在!")
        return

    print(f"开始处理目录: {directory_path}")

    for filename in os.listdir(directory_path):
        if not filename.endswith('.xml'):
            continue

        file_path = os.path.join(directory_path, filename)
        try:
            # 解析XML文件
            tree = ET.parse(file_path)
            root = tree.getroot()

            # 记录是否有修改
            file_modified = False
            file_replacements = 0

            # 查找所有name标签
            for name_tag in root.findall('.//object/name'):
                # 检查并替换标签文本
                if name_tag.text and 'black  ball' in name_tag.text:
                    old_text = name_tag.text
                    name_tag.text = old_text.replace('black  ball', 'black ball')
                    file_modified = True
                    file_replacements += 1

            # 如果文件被修改，保存更改
            if file_modified:
                tree.write(file_path, encoding='utf-8', xml_declaration=True)
                modified_files += 1
                total_replacements += file_replacements
                print(f'已修改: {filename} (替换了 {file_replacements} 处)')
            else:
                # 删除未修改的文件
                os.remove(file_path)
                print(f'已删除未修改的文件: {filename}')

        except Exception as e:
            print(f'处理文件 {filename} 时出错: {str(e)}')

    print(f'\n处理完成！')
    print(f'修改了 {modified_files} 个文件')
    print(f'总共替换了 {total_replacements} 处标签')

if __name__ == '__main__':
    annotations_dir = r'e:\src\yolo11\backup\Annotations'
    correct_xml_files(annotations_dir)