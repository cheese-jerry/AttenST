import argparse

def get_args():

    parser = argparse.ArgumentParser()

    # Parameter for style
    parser.add_argument('--without_init_adain', action='store_true', help="Disable initialization with AdaIN")
    parser.add_argument('--without_attn_injection', action='store_true', help="Disable attention injection")
    parser.add_argument('--layers', nargs='+', type=int, default=[7, 8, 9, 10, 11], help="Style-Guided Self-Attention Layers")

    # image path
    parser.add_argument('--cnt_path', type=str, default="/home/liyan/workspace/intern/huangbo/data/cnt/1.png", help="Path to content image")
    parser.add_argument('--sty_path', type=str, default="/home/liyan/workspace/intern/huangbo/data/sty/00.png", help="Path to style image")
    parser.add_argument('--save_dir', type=str, default='/home/liyan/workspace/intern/huangbo/Atten_ST_code/output', help="Directory to save results")

    cfg = parser.parse_args()
    return cfg

