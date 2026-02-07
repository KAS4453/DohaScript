#!/usr/bin/env python3
"""
Generate publication-quality figures from segmentation results
Run after line_segmentation_research.py completes
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set publication style
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")

# Configuration
RESULTS_FILE = "segmentation_results/segmentation_results_detailed.csv"
OUTPUT_FOLDER = "segmentation_results/figures"

def create_output_folder():
    """Create output folder for figures"""
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {OUTPUT_FOLDER}/")

def load_data():
    """Load segmentation results"""
    df = pd.read_csv(RESULTS_FILE)
    print(f"Loaded {len(df)} results from {RESULTS_FILE}")
    return df

def plot_score_distribution(df):
    """Figure 1: Distribution of segmentation scores"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram
    n, bins, patches = ax.hist(df['segmentation_score'], bins=20, 
                                edgecolor='black', alpha=0.7, color='steelblue')

    # Add difficulty thresholds
    ax.axvline(x=75, color='green', linestyle='--', linewidth=2, label='Easy threshold')
    ax.axvline(x=65, color='orange', linestyle='--', linewidth=2, label='Medium threshold')
    ax.axvline(x=45, color='red', linestyle='--', linewidth=2, label='Complex threshold')

    # Add mean line
    mean_score = df['segmentation_score'].mean()
    ax.axvline(x=mean_score, color='blue', linestyle='-', linewidth=2, 
              label=f'Mean: {mean_score:.2f}')

    ax.set_xlabel('Segmentation Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Segmentation Scores (N=531)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/fig1_score_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_FOLDER}/fig1_score_distribution.pdf", bbox_inches='tight')
    print("✓ Figure 1 saved: Score distribution")
    plt.close()

def plot_error_distribution(df):
    """Figure 2: Line detection error distribution"""
    fig, ax = plt.subplots(figsize=(8, 5))

    error_counts = df['line_difference'].value_counts().sort_index()
    colors = ['green' if e == 0 else 'orange' if e <= 2 else 'red' 
              for e in error_counts.index]

    bars = ax.bar(error_counts.index, error_counts.values, 
                  color=colors, edgecolor='black', alpha=0.7)

    # Add percentage labels on bars
    total = len(df)
    for bar, count in zip(bars, error_counts.values):
        height = bar.get_height()
        pct = (count / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Line Detection Error (|detected - 12|)', fontsize=12)
    ax.set_ylabel('Number of Documents', fontsize=12)
    ax.set_title('Distribution of Line Detection Errors', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add text box with accuracy metrics
    perfect = len(df[df['line_difference'] == 0])
    within_1 = len(df[df['line_difference'] <= 1])
    within_2 = len(df[df['line_difference'] <= 2])

    textstr = f'Perfect: {perfect/total*100:.1f}%\n'
    textstr += f'±1 line: {within_1/total*100:.1f}%\n'
    textstr += f'±2 lines: {within_2/total*100:.1f}%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/fig2_error_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_FOLDER}/fig2_error_distribution.pdf", bbox_inches='tight')
    print("✓ Figure 2 saved: Error distribution")
    plt.close()

def plot_method_comparison(df):
    """Figure 3: Method performance comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Box plot of scores by method
    methods = ['projection', 'contour', 'morphological']
    method_data = [df[df['method_used'] == m]['segmentation_score'].values 
                   for m in methods]

    bp = ax1.boxplot(method_data, labels=[m.capitalize() for m in methods],
                     patch_artist=True, showmeans=True)

    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_ylabel('Segmentation Score', fontsize=12)
    ax1.set_title('Score Distribution by Method', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add mean values as text
    for i, method in enumerate(methods):
        mean_val = df[df['method_used'] == method]['segmentation_score'].mean()
        ax1.text(i+1, 5, f'μ={mean_val:.1f}', ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Subplot 2: Method usage frequency
    method_counts = df['method_used'].value_counts()
    colors_pie = ['#ff9999', '#66b3ff', '#99ff99']

    wedges, texts, autotexts = ax2.pie(method_counts.values, 
                                        labels=[m.capitalize() for m in method_counts.index],
                                        autopct='%1.1f%%',
                                        colors=colors_pie,
                                        startangle=90)

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)
        autotext.set_weight('bold')

    ax2.set_title('Method Selection Frequency', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/fig3_method_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_FOLDER}/fig3_method_comparison.pdf", bbox_inches='tight')
    print("✓ Figure 3 saved: Method comparison")
    plt.close()

def plot_difficulty_analysis(df):
    """Figure 4: Difficulty distribution and characteristics"""
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

    # Subplot 1: Difficulty distribution
    ax1 = fig.add_subplot(gs[0, 0])
    diff_counts = df['difficulty'].value_counts().reindex(['Easy', 'Medium', 'Complex'])
    colors_diff = ['green', 'orange', 'red']

    bars = ax1.bar(diff_counts.index, diff_counts.values, color=colors_diff, 
                   alpha=0.7, edgecolor='black')

    for bar, count in zip(bars, diff_counts.values):
        height = bar.get_height()
        pct = (count / len(df)) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Number of Documents', fontsize=12)
    ax1.set_title('Difficulty Distribution', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Subplot 2: Score vs Line Difference scatter
    ax2 = fig.add_subplot(gs[0, 1])

    for difficulty, color in zip(['Easy', 'Medium', 'Complex'], colors_diff):
        mask = df['difficulty'] == difficulty
        ax2.scatter(df[mask]['line_difference'], 
                   df[mask]['segmentation_score'],
                   label=difficulty, alpha=0.6, s=30, color=color)

    ax2.set_xlabel('Line Detection Error', fontsize=12)
    ax2.set_ylabel('Segmentation Score', fontsize=12)
    ax2.set_title('Score vs Error by Difficulty', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Subplot 3: Mean metrics by difficulty
    ax3 = fig.add_subplot(gs[0, 2])

    metrics = ['line_height_variance', 'spacing_uniformity', 'coverage_ratio']
    difficulties = ['Easy', 'Medium', 'Complex']

    x = np.arange(len(metrics))
    width = 0.25

    for i, diff in enumerate(difficulties):
        means = [df[df['difficulty'] == diff][m].mean() for m in metrics]
        ax3.bar(x + i*width, means, width, label=diff, 
               color=colors_diff[i], alpha=0.7, edgecolor='black')

    ax3.set_xlabel('Metric', fontsize=12)
    ax3.set_ylabel('Mean Value', fontsize=12)
    ax3.set_title('Quality Metrics by Difficulty', fontsize=12, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(['Height Var', 'Spacing', 'Coverage'], rotation=15)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/fig4_difficulty_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_FOLDER}/fig4_difficulty_analysis.pdf", bbox_inches='tight')
    print("✓ Figure 4 saved: Difficulty analysis")
    plt.close()

def plot_correlation_heatmap(df):
    """Figure 5: Feature correlation heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))

    features = ['line_height_variance', 'spacing_uniformity', 'coverage_ratio',
                'line_straightness', 'segmentation_score', 'line_difference']

    corr_matrix = df[features].corr()

    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=12)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))

    labels = ['Height Var', 'Spacing', 'Coverage', 'Straightness', 'Score', 'Error']
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # Add correlation values
    for i in range(len(features)):
        for j in range(len(features)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/fig5_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_FOLDER}/fig5_correlation_heatmap.pdf", bbox_inches='tight')
    print("✓ Figure 5 saved: Correlation heatmap")
    plt.close()

def plot_processing_time(df):
    """Figure 6: Processing time analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Histogram of processing times
    ax1.hist(df['processing_time_ms'], bins=30, edgecolor='black', 
            alpha=0.7, color='skyblue')

    mean_time = df['processing_time_ms'].mean()
    median_time = df['processing_time_ms'].median()

    ax1.axvline(mean_time, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_time:.2f} ms')
    ax1.axvline(median_time, color='green', linestyle='--', linewidth=2,
               label=f'Median: {median_time:.2f} ms')

    ax1.set_xlabel('Processing Time (ms)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Processing Times', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Subplot 2: Processing time by method
    methods = ['projection', 'contour', 'morphological']
    time_data = [df[df['method_used'] == m]['processing_time_ms'].values 
                 for m in methods]

    bp = ax2.boxplot(time_data, labels=[m.capitalize() for m in methods],
                     patch_artist=True, showmeans=True)

    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax2.set_ylabel('Processing Time (ms)', fontsize=12)
    ax2.set_title('Processing Time by Method', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/fig6_processing_time.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_FOLDER}/fig6_processing_time.pdf", bbox_inches='tight')
    print("✓ Figure 6 saved: Processing time analysis")
    plt.close()

def generate_summary_figure(df):
    """Figure 7: Comprehensive summary figure for paper"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)

    # 1. Score distribution
    ax1 = fig.add_subplot(gs[0, :2])
    n, bins, patches = ax1.hist(df['segmentation_score'], bins=20, 
                                edgecolor='black', alpha=0.7, color='steelblue')
    mean_score = df['segmentation_score'].mean()
    ax1.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_score:.2f}')
    ax1.set_xlabel('Segmentation Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('(a) Score Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Difficulty pie chart
    ax2 = fig.add_subplot(gs[0, 2])
    diff_counts = df['difficulty'].value_counts().reindex(['Easy', 'Medium', 'Complex'])
    colors = ['green', 'orange', 'red']
    ax2.pie(diff_counts.values, labels=diff_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax2.set_title('(b) Difficulty Distribution', fontweight='bold')

    # 3. Error distribution
    ax3 = fig.add_subplot(gs[1, :2])
    error_counts = df['line_difference'].value_counts().sort_index()
    colors_bar = ['green' if e == 0 else 'orange' if e <= 2 else 'red' 
                  for e in error_counts.index]
    ax3.bar(error_counts.index, error_counts.values, color=colors_bar, 
           edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Line Detection Error')
    ax3.set_ylabel('Count')
    ax3.set_title('(c) Error Distribution', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # 4. Method comparison
    ax4 = fig.add_subplot(gs[1, 2])
    method_counts = df['method_used'].value_counts()
    ax4.bar(range(len(method_counts)), method_counts.values, 
           color=['#ff9999', '#66b3ff', '#99ff99'], edgecolor='black', alpha=0.7)
    ax4.set_xticks(range(len(method_counts)))
    ax4.set_xticklabels([m.capitalize() for m in method_counts.index], rotation=15)
    ax4.set_ylabel('Usage Count')
    ax4.set_title('(d) Method Selection', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # 5. Score vs error scatter
    ax5 = fig.add_subplot(gs[2, :2])
    colors_diff = {'Easy': 'green', 'Medium': 'orange', 'Complex': 'red'}
    for diff, color in colors_diff.items():
        mask = df['difficulty'] == diff
        ax5.scatter(df[mask]['line_difference'], df[mask]['segmentation_score'],
                   label=diff, alpha=0.6, s=20, color=color)
    ax5.set_xlabel('Line Detection Error')
    ax5.set_ylabel('Segmentation Score')
    ax5.set_title('(e) Score vs Error', fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # 6. Key statistics text
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    stats_text = f"""KEY STATISTICS (N={len(df)})

Perfect Segmentation:
{len(df[df['line_difference']==0])} ({len(df[df['line_difference']==0])/len(df)*100:.1f}%)

Mean Score:
{df['segmentation_score'].mean():.2f} ± {df['segmentation_score'].std():.2f}

Within ±1 line:
{len(df[df['line_difference']<=1])/len(df)*100:.1f}%

Within ±2 lines:
{len(df[df['line_difference']<=2])/len(df)*100:.1f}%

Avg Processing Time:
{df['processing_time_ms'].mean():.2f} ms
"""
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Comprehensive Segmentation Analysis Summary', 
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(f"{OUTPUT_FOLDER}/fig7_summary_combined.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_FOLDER}/fig7_summary_combined.pdf", bbox_inches='tight')
    print("✓ Figure 7 saved: Comprehensive summary")
    plt.close()

def main():
    """Generate all figures"""
    print("="*80)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*80)
    print()

    create_output_folder()
    df = load_data()

    print("\nGenerating figures...")
    plot_score_distribution(df)
    plot_error_distribution(df)
    plot_method_comparison(df)
    plot_difficulty_analysis(df)
    plot_correlation_heatmap(df)
    plot_processing_time(df)
    generate_summary_figure(df)

    print("\n" + "="*80)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\nLocation: {OUTPUT_FOLDER}/")
    print("\nFiles created:")
    print("  • fig1_score_distribution.png/pdf")
    print("  • fig2_error_distribution.png/pdf")
    print("  • fig3_method_comparison.png/pdf")
    print("  • fig4_difficulty_analysis.png/pdf")
    print("  • fig5_correlation_heatmap.png/pdf")
    print("  • fig6_processing_time.png/pdf")
    print("  • fig7_summary_combined.png/pdf")
    print("\nUse these figures in your research paper!")
    print("="*80)

if __name__ == "__main__":
    main()
