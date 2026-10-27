"""Run comparable, sequential CPU benchmarks on one recorded video.

Usage: python experiments/compare_cpu.py --video clip.mp4 --panorama pano.jpg
Two default runs have segmentation OFF; use --segmentation for actual YOLO.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panovideo.benchmark import BenchmarkConfig, compare_summaries, run_benchmark


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',required=True)
    parser.add_argument('--panorama',required=True)
    parser.add_argument('--output-dir',default='experiments/cpu_comparison')
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--max-frames',type=int,default=100)
    parser.add_argument('--segmentation',action='store_true')
    parser.add_argument('--model',default='yolo26n-seg.pt')
    args=parser.parse_args(argv)
    common=dict(video=args.video,panorama=args.panorama,output_dir=args.output_dir,
                warmup=args.warmup,max_frames=args.max_frames,
                segmentation=args.segmentation,model=args.model,device='cpu')
    runs=[run_benchmark(BenchmarkConfig(**common,threads=n,label=f'cpu_threads_{n}')) for n in (1,2)]
    csv_path=compare_summaries(runs,Path(args.output_dir)/'comparison.csv')
    print('Comparison:',csv_path)
    for r in runs:
        m=r['metrics']['total_processing_ms']
        print(f"{r['label']}: frames={r['frames']}, median={m['median']:.3f}ms, mean={m['mean']:.3f}ms, p95={m['p95']:.3f}ms; throughput from mean processing time={1000/m['mean'] if m['mean'] else 0:.2f} fps")


if __name__=='__main__':
    main()
