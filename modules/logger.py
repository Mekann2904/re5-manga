import logging
import sys

def setup_logger(level=logging.INFO):
    """
    ロガーを設定する関数
    """
    logger = logging.getLogger('manga_converter')
    logger.setLevel(level)

    # すでにハンドラが設定されている場合は追加しない
    if not logger.handlers:
        # コンソールハンドラ
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# デフォルトロガー
logger = setup_logger()
