import argparse
import os
import glob
import cv2
import logging
import numpy as np # Added for np.full

from modules.logger import setup_logger, logger
from modules.io import load_image, save_image
from modules.preprocess import resize, adjust_contrast
from modules.edge import extract_canny, extract_highpass, extract_binary, combine_edges, dilate_edges
from modules.tone import generate_posterized, generate_levels, apply_mask, generate_halftone, make_dot_pattern, make_grid_pattern, make_stripe_pattern, make_speed_lines
from modules.postprocess import remove_noise, cleanup_edges
from modules.utils import get_output_path

def adjust_shadow_highlight(img_gray, shadow=50, highlight=0):
    # シャドウ部分を明るくする（参考サイトのシャドウ・ハイライト調整の簡易版）
    # shadow: 0-100, highlight: 0-100
    img_float = img_gray.astype(np.float32)
    # シャドウ補正
    img_float = img_float + (shadow / 100.0) * (255 - img_float)
    # ハイライト補正（今回は未使用）
    img_float = np.clip(img_float, 0, 255)
    return img_float.astype(np.uint8)

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
        # --- 色調補正（シャドウ明るく） ---
        img_gray = adjust_shadow_highlight(img_gray, shadow=50, highlight=0)
        # --- 平滑化 ---
        img_blur = cv2.bilateralFilter(img_gray, d=9, sigmaColor=75, sigmaSpace=75)

        # 3. 線画抽出（3種）
        # 輪郭検出（強化オプション対応）
        if args.edge_enhance:
            # 輪郭線強化モード
            edge_input = img_gray.copy()
            if args.edge_blur > 0:
                # 前処理ブラーでノイズ除去
                edge_input = cv2.GaussianBlur(edge_input, (args.edge_blur*2+1, args.edge_blur*2+1), 0)
            
            # より敏感なCanny設定
            enhanced_canny1 = max(10, int(args.canny1 * 0.7))
            enhanced_canny2 = min(200, int(args.canny2 * 1.3))
            edge_canny = extract_canny(edge_input, enhanced_canny1, enhanced_canny2)
            
            # 輪郭線強度調整
            if args.edge_strength != 1.0:
                edge_canny = cv2.multiply(edge_canny, args.edge_strength)
                edge_canny = np.clip(edge_canny, 0, 255).astype(np.uint8)
        else:
            # 通常モード
            edge_canny = extract_canny(img_blur, args.canny1, args.canny2)
        
        edge_canny_bin = cv2.threshold(edge_canny, 170, 255, cv2.THRESH_BINARY)[1]
        # ハイパス（強度調整可能）
        edge_highpass = extract_highpass(img_resized, args.hp_kernel, args.hp_thresh)
        # ハイパスフィルタの強度を調整
        if args.hp_strength != 1.0:
            edge_highpass = cv2.multiply(edge_highpass, args.hp_strength)
            edge_highpass = np.clip(edge_highpass, 0, 255).astype(np.uint8)
        edge_highpass_bin = cv2.threshold(edge_highpass, 100, 255, cv2.THRESH_BINARY)[1]
        # ベタ塗りレイヤーの生成を自動しきい値＋コントラスト強調で
        img_for_binary = cv2.equalizeHist(img_blur)
        auto_thresh = int(np.percentile(img_for_binary, 30))
        edge_binary = extract_binary(img_for_binary, auto_thresh)
        # --- デバッグ用画像の保存 ---
        if args.debug:
            save_image(edge_canny, get_output_path(args.input, "debug_canny.png"), args.dpi)
            save_image(edge_highpass, get_output_path(args.input, "debug_highpass.png"), args.dpi)
            save_image(edge_binary, get_output_path(args.input, "debug_binary.png"), args.dpi)

        # 3つの線画レイヤーをbitwise_orで合成（線を統合する）
        line_art = cv2.bitwise_or(edge_canny_bin, edge_highpass_bin)
        line_art = cv2.bitwise_or(line_art, edge_binary)
        
        # 輪郭線強化モードの場合、追加処理
        if args.edge_enhance:
            from modules.edge import enhance_edges
            enhanced_edges = enhance_edges(img_gray, args.edge_strength)
            enhanced_edges_bin = cv2.threshold(enhanced_edges, 50, 255, cv2.THRESH_BINARY)[1]
            line_art = cv2.bitwise_or(line_art, enhanced_edges_bin)
            if args.debug:
                save_image(enhanced_edges, get_output_path(args.input, "debug_enhanced_edges.png"), args.dpi)
        
        if args.debug:
            save_image(line_art, get_output_path(args.input, "debug_lineart_after_combine.png"), args.dpi)

        # 線を太くするため、一時的に白黒反転（線が白、背景が黒）
        line_art_inv = cv2.bitwise_not(line_art)

        # 膨張処理で線を太くする
        dilated_inv = dilate_edges(line_art_inv, kernel_size=2, iterations=1)
        
        # 再度反転し、最終的な線画（線が黒、背景が白）を得る
        line_art_final = cv2.bitwise_not(dilated_inv)
        if args.debug:
            save_image(line_art_final, get_output_path(args.input, "debug_lineart_after_dilate.png"), args.dpi)

        # 2値化でクッキリさせる
        _, line_art_bin = cv2.threshold(line_art_final, 200, 255, cv2.THRESH_BINARY)

        # 線画だけの画像を保存
        output_line_path = get_output_path(args.input, args.output)
        output_line_path = output_line_path.replace('.png', '_line.png').replace('.jpg', '_line.png')
        save_image(line_art_bin, output_line_path, args.dpi)

        # 4. トーン生成
        if args.tone_type == 'grayscale':
            from modules.tone import generate_grayscale_tone
            # ベースとなるグレースケール画像を準備
            tone_base = cv2.GaussianBlur(img_gray, (5,5), 0)  # ノイズ除去
            # グレースケールトーンを生成（階調数を指定）
            tone = generate_grayscale_tone(tone_base, levels=args.tone_levels)
        else:
            # 従来のハーフトーン処理
            tone_poster = generate_posterized(img_gray, 4)
            tone_simple = cv2.GaussianBlur(img_gray, (3,3), 0)
            tone = cv2.addWeighted(tone_poster, 0.5, tone_simple, 0.5, 0)
            tone = generate_halftone(tone, dot_size=3)

        # 5. 線画をトーンに合成
        # line_art_binは線が黒(0)なので、黒の部分をマスクとして利用
        final_image = tone.copy()
        final_image[line_art_bin == 0] = 0

        # --- 集中線合成 ---
        if args.speed_line:
            speed = make_speed_lines(final_image.shape, num_lines=80)
            final_image = cv2.bitwise_and(final_image, speed)

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
    parser.add_argument('--edge-enhance', action='store_true', help='輪郭線強化モードを有効化')
    parser.add_argument('--edge-strength', type=float, default=1.0, help='輪郭線強度（1.0-3.0）')
    parser.add_argument('--edge-blur', type=int, default=1, help='輪郭線前処理ブラー（0で無効）')
    parser.add_argument('--hp-kernel', type=int, default=5, help='ハイパスフィルタカーネルサイズ')
    parser.add_argument('--hp-thresh', type=int, default=10, help='ハイパスフィルタ閾値')
    parser.add_argument('--hp-strength', type=float, default=1.0, help='ハイパスフィルタ強度（1.0-3.0）')
    parser.add_argument('--thresh', type=int, default=128, help='二値化閾値')
    parser.add_argument('--poster-levels', type=int, default=4, help='ポスタリゼーション階調数')
    parser.add_argument('--mask-thresh', type=int, default=200, help='マスク二値化閾値')
    parser.add_argument('--batch', action='store_true', help='フォルダ単位でバッチ処理')
    parser.add_argument('--debug', action='store_true', help='デバッグログを有効化')
    parser.add_argument('--dilate-kernel', type=int, default=2, help='線画膨張カーネルサイズ')
    parser.add_argument('--dilate-iter', type=int, default=1, help='線画膨張回数')
    parser.add_argument('--tone-type', type=str, default='grayscale', choices=['grayscale', 'halftone', 'dot', 'grid', 'stripe'], help='トーン種別')
    parser.add_argument('--tone-levels', type=int, default=6, help='グレースケールトーンの階調数')
    parser.add_argument('--speed-line', action='store_true', help='集中線（スピード線）を背景に合成')

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
