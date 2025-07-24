import argparse
import os
import glob
import cv2

from modules.logger import setup_logger, logger
from modules.io import load_image, save_image
from modules.preprocess import resize, adjust_contrast
from modules.edge import extract_canny, extract_highpass, extract_binary, combine_edges, dilate_edges
from modules.tone import generate_posterized, generate_levels, apply_mask, generate_halftone
from modules.postprocess import remove_noise, cleanup_edges
from modules.utils import get_output_path

def process_image(args):
    """
    単一の画像を処理する
    """
    try:
        # 1. 画像読み込み
        img = load_image(args.input)

        # 2. 前処理
        img_resized = resize(img, args.dpi)
        img_contrast = adjust_contrast(img_resized)
        img_gray = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2GRAY)

        # 3. 線画抽出
        edge_canny = extract_canny(img_gray, args.canny1, args.canny2)
        edge_highpass = extract_highpass(img_resized, args.hp_kernel, 10) # 閾値は仮
        edge_binary = extract_binary(img_gray, args.thresh)
        
        # 線画合成
        line_art = combine_edges([edge_canny, edge_binary])
        # --- 追加: 線画を太くする ---
        line_art = dilate_edges(line_art, kernel_size=2, iterations=1)

        # 4. トーン生成
        # --- 追加: 網点トーンを生成 ---
        tone_halftone = generate_halftone(img_gray, dot_size=4)
        # tone_posterized = generate_posterized(img_gray, args.poster_levels)
        
        # 5. トーン合成 (線画と網点トーンを合成)
        final_image = cv2.bitwise_and(line_art, tone_halftone)

        # 6. 後処理
        final_image = remove_noise(final_image)
        final_image = cleanup_edges(final_image)

        # 7. 画像保存
        output_path = get_output_path(args.input, args.output)
        save_image(final_image, output_path, args.dpi)
        logger.info(f"Successfully processed {args.input}")

    except Exception as e:
        logger.error(f"Failed to process {args.input}: {e}")

def main():
    parser = argparse.ArgumentParser(description='写真→白黒マンガ背景変換ツール')
    parser.add_argument('input', help='入力ファイルまたはディレクトリ')
    parser.add_argument('output', nargs='?', help='出力ファイルまたはディレクトリ')
    parser.add_argument('-d', '--dpi', type=int, default=1200, help='出力解像度')
    parser.add_argument('--canny1', type=int, default=50, help='Canny閾値1')
    parser.add_argument('--canny2', type=int, default=150, help='Canny閾値2')
    parser.add_argument('--hp-kernel', type=int, default=5, help='ハイパスフィルタカーネルサイズ')
    parser.add_argument('--thresh', type=int, default=128, help='二値化閾値')
    parser.add_argument('--poster-levels', type=int, default=4, help='ポスタリゼーション階調数')
    parser.add_argument('--mask-thresh', type=int, default=200, help='マスク二値化閾値')
    parser.add_argument('--batch', action='store_true', help='フォルダ単位でバッチ処理')
    parser.add_argument('--debug', action='store_true', help='デバッグログを有効化')

    args = parser.parse_args()

    # ロガー設定
    if args.debug:
        setup_logger(level=logging.DEBUG)
    
    logger.info("Starting manga background conversion process.")

    if args.batch:
        if not os.path.isdir(args.input):
            logger.error("Batch mode requires an input directory.")
            return
        
        output_dir = args.output if args.output and os.path.isdir(args.output) else args.input
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_files = glob.glob(os.path.join(args.input, '*.jpg')) + \
                      glob.glob(os.path.join(args.input, '*.png'))

        for file_path in image_files:
            # バッチ処理用に引数を変更
            batch_args = argparse.Namespace(**vars(args))
            batch_args.input = file_path
            batch_args.output = os.path.join(output_dir, os.path.basename(file_path))
            process_image(batch_args)
    else:
        process_image(args)

    logger.info("Process finished.")

if __name__ == '__main__':
    main()
