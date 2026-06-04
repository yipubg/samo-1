"""
VCTK 测试集批量构建工具
=======================
本脚本用于从 VCTK-VC 数据集批量构建标准化的测试集。

背景:
  VCTK-VC 数据集包含 1 种真实语音 (clean) 和 4 种伪造方法:
    0-VCTK-Corpus3/   → 真实语音 (clean)
    1-AdaIN-VC/       → 伪造方法 1: AdaIN 风格迁移语音转换
    2-CycleGAN-VC/    → 伪造方法 2: CycleGAN 语音转换
    3-VQVC/           → 伪造方法 3: Vector Quantization 语音转换
    4-VQVC+/          → 伪造方法 4: 改进版 VQVC

工作原理:
  1. 从 clean 目录中随机抽取 N 条音频（默认 1500 条）
  2. 对每种伪造方法，复制对应的 clean + spoof 音频到独立测试目录
  3. 输出 4 个独立的测试集，每个包含 clean/ 和 spoof/ 子目录

输出结构:
  test_sets/
    test_adain/       (clean/ + spoof/)
    test_cyclegan/    (clean/ + spoof/)
    test_vqvc/        (clean/ + spoof/)
    test_vqvc_plus/   (clean/ + spoof/)

使用示例:
  python build_test_sets.py --vctk_root "D:\\testone\\VCTK-VC" --output_root ./test_sets --num 1500
"""

import os
import shutil
import random
import argparse


def get_relative_wav_paths(root_dir):
    """
    递归扫描目录，获取所有 WAV 文件的相对路径（相对于 root_dir）。

    参数:
        root_dir: 要扫描的根目录

    返回:
        rel_paths: 排序后的相对路径列表
    """
    rel_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.wav'):
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, root_dir)
                rel_paths.append(rel)
    return sorted(rel_paths)


def copy_by_rel_paths(src_root, dst_root, rel_paths):
    """
    根据相对路径列表，从源目录复制文件到目标目录，保持目录结构。

    参数:
        src_root:  源根目录
        dst_root:  目标根目录
        rel_paths: 要复制的文件相对路径列表

    返回:
        copied: 成功复制的文件数
        missing: 源目录中不存在的文件列表
    """
    copied = 0
    missing = []
    for rel in rel_paths:
        src_file = os.path.join(src_root, rel)
        dst_file = os.path.join(dst_root, rel)
        if os.path.exists(src_file):
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy2(src_file, dst_file)  # copy2 保留文件元数据
            copied += 1
        else:
            missing.append(rel)
    return copied, missing


def main():
    parser = argparse.ArgumentParser(
        description='批量构建VCTK测试集: 4种伪造方法共享相同的1500条clean音频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python build_test_sets.py --vctk_root "D:\\testone\\VCTK-VC" --output_root ./test_sets --num 1500

目录结构预期:
  VCTK-VC/
    0-VCTK-Corpus3/   (clean - 真实语音)
    1-AdaIN-VC/       (spoof - 伪造方法1)
    2-CycleGAN-VC/    (spoof - 伪造方法2)
    3-VQVC/           (spoof - 伪造方法3)
    4-VQVC+/          (spoof - 伪造方法4)

输出结构:
  test_sets/
    test_adain/       (clean/ + spoof/)
    test_cyclegan/    (clean/ + spoof/)
    test_vqvc/        (clean/ + spoof/)
    test_vqvc_plus/   (clean/ + spoof/)
        """
    )

    parser.add_argument('--vctk_root', type=str, required=True,
                        help='VCTK-VC 根目录（包含 0-VCTK-Corpus3, 1-AdaIN-VC 等子目录）')
    parser.add_argument('--output_root', type=str, default='./test_sets',
                        help='输出根目录（默认 ./test_sets）')
    parser.add_argument('--num', type=int, default=1500,
                        help='每种类型抽取的音频数量（默认 1500）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（保证可复现）')

    args = parser.parse_args()
    random.seed(args.seed)

    # ---- 定义输入目录 ----
    clean_src = os.path.join(args.vctk_root, '0-VCTK-Corpus3')

    # 伪造方法映射: {方法名: (伪造源目录, 输出子目录名)}
    methods = {
        'adain': ('1-AdaIN-VC', 'test_adain'),
        'cyclegan': ('2-CycleGAN-VC', 'test_cyclegan'),
        'vqvc': ('3-VQVC', 'test_vqvc'),
        'vqvc_plus': ('4-VQVC+', 'test_vqvc_plus'),
    }

    # ---- 步骤1: 扫描 clean 目录，随机抽取音频 ----
    print(f"Scanning clean audio: {clean_src}")
    all_clean_rel = get_relative_wav_paths(clean_src)
    print(f"  Total clean files: {len(all_clean_rel)}")

    if len(all_clean_rel) < args.num:
        print(f"  Warning: 只有 {len(all_clean_rel)} 条可用，将使用全部")
        selected_rel = all_clean_rel
    else:
        selected_rel = random.sample(all_clean_rel, args.num)
        selected_rel = sorted(selected_rel)

    print(f"  Selected {len(selected_rel)} clean files (shared across all methods)\n")

    # ---- 步骤2: 为每种伪造方法构建测试集 ----
    results = {}
    for method_name, (spoof_dir_name, test_dir_name) in methods.items():
        spoof_src = os.path.join(args.vctk_root, spoof_dir_name)
        test_dir = os.path.join(args.output_root, test_dir_name)
        clean_dst = os.path.join(test_dir, 'clean')
        spoof_dst = os.path.join(test_dir, 'spoof')

        print(f"{'=' * 60}")
        print(f"Building: {method_name.upper()}")
        print(f"  Spoof source: {spoof_src}")
        print(f"  Output: {test_dir}")

        # 检查伪造源目录是否存在
        if not os.path.exists(spoof_src):
            print(f"  Missing spoof source, skipping!")
            results[method_name] = 'skipped (source missing)'
            continue

        # 清理旧输出
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        # 复制 clean 音频（所有方法共享相同的 clean 集）
        n_clean, miss_clean = copy_by_rel_paths(clean_src, clean_dst, selected_rel)

        # 在伪造目录中匹配对应文件
        spoof_available = get_relative_wav_paths(spoof_src)
        spoof_set = set(spoof_available)

        matched_spoof_rel = [rel for rel in selected_rel if rel in spoof_set]
        missed_spoof_rel = [rel for rel in selected_rel if rel not in spoof_set]

        n_spoof, miss_spoof = copy_by_rel_paths(spoof_src, spoof_dst, matched_spoof_rel)

        # 如果匹配到的 spoof 不足 num 条，从伪造目录剩余文件中补足
        if n_spoof < args.num:
            remaining = [rel for rel in spoof_available if rel not in matched_spoof_rel]
            need = args.num - n_spoof
            extra = random.sample(remaining, min(need, len(remaining))) if remaining else []
            n_extra, _ = copy_by_rel_paths(spoof_src, spoof_dst, extra)
            n_spoof += n_extra
            print(f"  Matched {len(matched_spoof_rel)} corresponding spoof files")
            print(f"  Added {n_extra} extra files to reach {n_spoof}")
        else:
            n_extra = 0

        # 验证最终结果
        final_clean = get_relative_wav_paths(clean_dst)
        final_spoof = get_relative_wav_paths(spoof_dst)

        print(f"  Clean: {len(final_clean)} files")
        print(f"  Spoof: {len(final_spoof)} files")
        if missed_spoof_rel:
            print(f"  {len(missed_spoof_rel)} clean files had no corresponding spoof")

        results[method_name] = {
            'clean': len(final_clean),
            'spoof': len(final_spoof),
            'output': test_dir
        }

    # ---- 步骤3: 输出汇总信息 ----
    print(f"\n{'=' * 60}")
    print(f"所有测试集构建完成")
    print(f"{'=' * 60}")
    for method, info in results.items():
        if isinstance(info, dict):
            print(f"  {method:12s}: clean={info['clean']:4d}, spoof={info['spoof']:4d}  ->  {info['output']}")
        else:
            print(f"  {method:12s}: {info}")

    print(f"\n测试命令示例:")
    print(f"  python test_vctk_samo.py -m ./models/samo.pt --loss_type samo --samo_mode with_enroll \\")
    print(f"    --vctk_clean \"{os.path.join(args.output_root, 'test_adain', 'clean')}\" \\")
    print(f"    --vctk_attack \"{os.path.join(args.output_root, 'test_adain', 'spoof')}\" \\")
    print(f"    -o results_adain.txt")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
