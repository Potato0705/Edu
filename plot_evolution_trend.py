
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
    
    # Global Font Settings
    plt.rcParams.update({'font.size': args.fontsize})
    plt.rcParams.update({'font.family': args.fontfamily})

    # Plot Line
    plt.plot(grand_means['generation'], grand_means['qwk'], 
             marker=args.marker, 
             linestyle=args.linestyle, 
             linewidth=args.linewidth, 
             color=args.color, 
             label='Average QWK (All Prompts)')

    # Labels and Title
    plt.title(args.title, fontsize=args.title_fontsize, fontweight='bold', pad=20)
    plt.xlabel(args.xlabel, fontsize=args.label_fontsize)
    plt.ylabel(args.ylabel, fontsize=args.label_fontsize)
    
    # Ticks
    plt.tick_params(axis='both', which='major', labelsize=args.tick_fontsize)
    
    # Grid
    if args.grid:
        plt.grid(True, linestyle='--', alpha=0.7)

    # Legend
    plt.legend(fontsize=args.legend_fontsize)

    # Adjust layout
    plt.tight_layout()

    # Save
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

    # Print Data Table
    print("\nGeneration | Grand Mean QWK")
    print("-" * 30)
    for _, row in grand_means.iterrows():
        print(f"{int(row['generation']):<10} | {row['qwk']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Plot Evolution Trend from CSV")
    
    parser.add_argument("--csv", type=str, default="evolution_trend_data.csv", help="Input CSV file path")
    parser.add_argument("--output", type=str, default="evolution_trend.png", help="Output image file path")
    
    # Content Params
    parser.add_argument("--title", type=str, default="Evolution of Test QWK (Average of 8 Prompts)", help="Chart Title")
    parser.add_argument("--xlabel", type=str, default="Generation", help="X Axis Label")
    parser.add_argument("--ylabel", type=str, default="Test QWK", help="Y Axis Label")
    
    # Style Params
    parser.add_argument("--color", type=str, default="#1f77b4", help="Line color (hex or name)")
    parser.add_argument("--marker", type=str, default="o", help="Marker style (o, s, ^, etc)")
    parser.add_argument("--linestyle", type=str, default="-", help="Line style (-, --, :, etc)")
    parser.add_argument("--linewidth", type=float, default=2.0, help="Line width")
    
    parser.add_argument("--width", type=float, default=10.0, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=6.0, help="Figure height in inches")
    parser.add_argument("--dpi", type=int, default=100, help="Figure DPI")
    
    parser.add_argument("--grid", action="store_true", default=True, help="Show grid")
    parser.add_argument("--no-grid", action="store_false", dest="grid", help="Hide grid")
    
    # Font Params
    parser.add_argument("--fontfamily", type=str, default="sans-serif", help="Font family")
    parser.add_argument("--fontsize", type=float, default=12, help="Base font size")
    parser.add_argument("--title_fontsize", type=float, default=16, help="Title font size")
    parser.add_argument("--label_fontsize", type=float, default=14, help="Axis label font size")
    parser.add_argument("--tick_fontsize", type=float, default=12, help="Tick label font size")
    parser.add_argument("--legend_fontsize", type=float, default=12, help="Legend font size")

    args = parser.parse_args()
    plot_evolution(args)

if __name__ == "__main__":
    main()
