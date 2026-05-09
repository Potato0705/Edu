
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_evolution(args):
    if not os.path.exists(args.csv):
        print(f"Error: File {args.csv} not found.")
        return

    print(f"Reading data from {args.csv}...")
    df = pd.read_csv(args.csv)

    if df.empty:
        print("Error: CSV is empty.")
        return

    # Aggregate data
    # Group by [generation, prompt] -> mean(qwk) over folds
    prompt_gen_means = df.groupby(['generation', 'prompt'])['qwk'].mean().reset_index()
    
    # Group by [generation] -> mean(mean_qwk) over prompts (Grand Mean)
    grand_means = prompt_gen_means.groupby('generation')['qwk'].mean().reset_index()
    grand_means = grand_means.sort_values('generation')

    # Setup Plot
    plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
    
    # Global Font Settings for Academic Style
    plt.rcParams['font.family'] = args.fontfamily
    if args.fontfamily == 'serif':
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    elif args.fontfamily == 'sans-serif':
        plt.rcParams['font.sans-serif'] = ['Arial'] + plt.rcParams['font.sans-serif']
    
    plt.rcParams.update({'font.size': args.fontsize})
    plt.rcParams['axes.unicode_minus'] = False # Fix for minus signs

    # Setup Plot
    plt.figure(figsize=(args.width, args.height)) # SVG doesn't strictly need DPI, but good for display

    # Plot Line with Academic Style
    plt.plot(grand_means['generation'], grand_means['qwk'], 
             marker=args.marker, 
             markersize=8, # Slightly larger markers
             linestyle=args.linestyle, 
             linewidth=args.linewidth, 
             color=args.color, 
             label='Average QWK')

    # Plot Baseline
    if args.baseline is not None:
        plt.axhline(y=args.baseline, color='#d62728', linestyle='--', linewidth=args.linewidth, label=args.baseline_label)

    # Labels and Title
    # Academic charts often generally don't utilize a top title (captions are used instead).
    # But we keep it if provided.
    if args.title:
        plt.title(args.title, fontsize=args.title_fontsize, fontweight='bold', pad=15)
        
    plt.xlabel(args.xlabel, fontsize=args.label_fontsize, fontweight='bold')
    plt.ylabel(args.ylabel, fontsize=args.label_fontsize, fontweight='bold')
    
    # Ticks formatting
    plt.tick_params(axis='both', which='major', labelsize=args.tick_fontsize, direction='in', width=1.5, length=6)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', direction='in', width=1, length=3)
    
    # Grid (Academic often prefers no grid or very subtle)
    if args.grid:
        plt.grid(True, linestyle=':', alpha=0.5, color='gray')

    # Spines (Make them thicker like academic plots)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Legend
    plt.legend(fontsize=args.legend_fontsize, frameon=False) # No box is cleaner

    # Adjust layout
    plt.tight_layout()

    # Save
    plt.savefig(args.output, format='svg', transparent=True, bbox_inches='tight')
    print(f"Plot saved to {args.output}")

    # Print Data Table
    print("\nGeneration | Grand Mean QWK")
    print("-" * 30)
    for _, row in grand_means.iterrows():
        print(f"{int(row['generation']):<10} | {row['qwk']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Plot Evolution Trend from CSV")
    
    parser.add_argument("--csv", type=str, default="evolution_trend_data.csv", help="Input CSV file path")
    parser.add_argument("--output", type=str, default="evolution_trend.svg", help="Output image file path (SVG recommended)")
    
    # Content Params
    parser.add_argument("--title", type=str, default="", help="Chart Title (Leave empty for academic papers)")
    parser.add_argument("--xlabel", type=str, default="Generation", help="X Axis Label")
    parser.add_argument("--ylabel", type=str, default="Average QWK", help="Y Axis Label")
    parser.add_argument("--baseline", type=float, default=None, help="Baseline QWK value for horizontal line")
    parser.add_argument("--baseline_label", type=str, default="No Induction Baseline", help="Label for baseline")
    
    # Style Params
    parser.add_argument("--color", type=str, default="#1f77b4", help="Line color")
    parser.add_argument("--marker", type=str, default="o", help="Marker style")
    parser.add_argument("--linestyle", type=str, default="-", help="Line style")
    parser.add_argument("--linewidth", type=float, default=2.5, help="Line width (thicker for readability)")
    
    parser.add_argument("--width", type=float, default=8.0, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=6.0, help="Figure height in inches")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    
    parser.add_argument("--grid", action="store_true", default=True, help="Show grid")
    parser.add_argument("--no-grid", action="store_false", dest="grid", help="Hide grid")
    
    # Font Params
    parser.add_argument("--fontfamily", type=str, default="serif", help="Font family (serif -> Times New Roman)")
    parser.add_argument("--fontsize", type=float, default=18, help="Base font size")
    parser.add_argument("--title_fontsize", type=float, default=24, help="Title font size")
    parser.add_argument("--label_fontsize", type=float, default=22, help="Axis label font size")
    parser.add_argument("--tick_fontsize", type=float, default=20, help="Tick label font size")
    parser.add_argument("--legend_fontsize", type=float, default=20, help="Legend font size")

    args = parser.parse_args()
    plot_evolution(args)

if __name__ == "__main__":
    main()
