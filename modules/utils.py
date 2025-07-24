import os

def get_output_path(input_path, output_path=None, suffix='_manga'):
    """
    出力ファイルパスを生成する
    """
    if output_path:
        return output_path
    
    base, ext = os.path.splitext(input_path)
    return f"{base}{suffix}{ext}"
