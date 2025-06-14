import os
import py7zr
import zipfile
import rarfile
import shutil
from pathlib import Path

def extract_archive(file_path, extract_dir):
    """尝试使用不同的解压方法解压文件"""
    try:
        # 尝试作为7z文件解压
        with py7zr.SevenZipFile(file_path, mode='r') as z:
            z.extractall(extract_dir)
            return True
    except Exception:
        try:
            # 尝试作为zip文件解压
            with zipfile.ZipFile(file_path, 'r') as z:
                z.extractall(extract_dir)
                return True
        except Exception:
            try:
                # 尝试作为rar文件解压
                with rarfile.RarFile(file_path, 'r') as z:
                    z.extractall(extract_dir)
                    return True
            except Exception as e:
                print(f"无法解压文件 {file_path.name}: {str(e)}")
                return False

def extract_and_organize():
    # 设置路径
    current_dir = Path(__file__).parent
    excel_output_dir = current_dir / "extracted_excel_files"
    
    # 创建输出目录（如果不存在）
    excel_output_dir.mkdir(exist_ok=True)
    
    # 遍历当前目录下的所有压缩文件
    for file in current_dir.glob("*.*"):
        if file.suffix.lower() not in ['.7z', '.zip', '.rar']:
            continue
            
        print(f"\n正在处理: {file.name}")
        
        # 创建临时解压目录
        temp_dir = current_dir / f"temp_{file.stem}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # 尝试解压文件
            if not extract_archive(file, temp_dir):
                print(f"警告: 无法解压 {file.name}，跳过处理")
                continue
            
            # 查找并移动所有.xlsx文件
            excel_files = list(temp_dir.rglob("*.xlsx"))
            if not excel_files:
                print(f"警告: {file.name} 中没有找到Excel文件")
            
            for excel_file in excel_files:
                # 构建目标路径
                target_path = excel_output_dir / excel_file.name
                
                # 如果目标文件已存在，添加数字后缀
                counter = 1
                while target_path.exists():
                    target_path = excel_output_dir / f"{excel_file.stem}_{counter}{excel_file.suffix}"
                    counter += 1
                
                # 移动文件
                shutil.move(str(excel_file), str(target_path))
                print(f"已移动: {excel_file.name} -> {target_path.name}")
        
        except Exception as e:
            print(f"处理 {file.name} 时出错: {str(e)}")
        
        finally:
            # 清理临时目录
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    print("\n处理完成！")
    print(f"所有Excel文件已移动到: {excel_output_dir}")

if __name__ == "__main__":
    extract_and_organize()
