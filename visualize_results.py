#!/usr/bin/env python3
"""
Group Behavior Visualization Tool
Creates various charts and graphs from the enhanced CSV analysis data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GroupBehaviorVisualizer:
    def __init__(self, csv_file_path):
        """Initialize visualizer with enhanced CSV file path"""
        self.csv_file_path = csv_file_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load the enhanced CSV data"""
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"Enhanced CSV file not found: {self.csv_file_path}")
        
        self.df = pd.read_csv(self.csv_file_path)
        print(f"✅ Loaded {len(self.df)} records for visualization")
    
    def create_timeline_visualization(self, save_path="output/timeline_analysis.png"):
        """Create a timeline showing group formations and dispersals over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Get formation and dispersal events
        formations = self.df[self.df['is_formation'] == True]
        dispersals = self.df[self.df['is_dispersal'] == True]
        
        # Plot 1: Group formations over time
        ax1.scatter(formations['time_seconds'], formations['group_id'], 
                   c='green', s=100, alpha=0.7, label='Group Formations')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Group ID')
        ax1.set_title('Group Formations Timeline', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Group dispersals over time
        ax2.scatter(dispersals['time_seconds'], dispersals['group_id'], 
                   c='red', s=100, alpha=0.7, label='Group Dispersals')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Group ID')
        ax2.set_title('Group Dispersals Timeline', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Timeline visualization saved: {save_path}")
    
    def create_dwell_time_analysis(self, save_path="output/dwell_time_analysis.png"):
        """Create visualizations for dwell time patterns"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Filter out zero dwell times (formations)
        dwell_data = self.df[self.df['dwell_time_seconds'] > 0]
        
        # Plot 1: Dwell time distribution histogram
        ax1.hist(dwell_data['dwell_time_seconds'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Dwell Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Dwell Time Distribution', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plot of dwell times by group
        group_dwell = dwell_data.groupby('group_id')['dwell_time_seconds'].mean().reset_index()
        ax2.bar(group_dwell['group_id'], group_dwell['dwell_time_seconds'], 
               color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Group ID')
        ax2.set_ylabel('Average Dwell Time (seconds)')
        ax2.set_title('Average Dwell Time by Group', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Dwell time vs group size
        size_dwell = self.df.groupby('member_count')['dwell_time_seconds'].mean().reset_index()
        ax3.scatter(size_dwell['member_count'], size_dwell['dwell_time_seconds'], 
                   s=100, alpha=0.7, color='purple')
        ax3.set_xlabel('Group Size (number of people)')
        ax3.set_ylabel('Average Dwell Time (seconds)')
        ax3.set_title('Dwell Time vs Group Size', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative dwell time over time
        cumulative_dwell = self.df['dwell_time_seconds'].cumsum()
        ax4.plot(self.df['time_seconds'], cumulative_dwell, linewidth=2, color='orange')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Cumulative Dwell Time (seconds)')
        ax4.set_title('Cumulative Dwell Time Over Time', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Dwell time analysis saved: {save_path}")
    
    def create_group_size_analysis(self, save_path="output/group_size_analysis.png"):
        """Create visualizations for group size patterns"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Group size distribution pie chart
        size_counts = self.df['member_count'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(size_counts)))
        ax1.pie(size_counts.values, labels=size_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax1.set_title('Group Size Distribution', fontweight='bold')
        
        # Plot 2: Group size over time
        ax2.scatter(self.df['time_seconds'], self.df['member_count'], 
                   alpha=0.6, c=self.df['group_id'], cmap='tab10')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Group Size')
        ax2.set_title('Group Size Over Time', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Group size frequency bar chart
        size_freq = self.df['member_count'].value_counts().sort_index()
        ax3.bar(size_freq.index, size_freq.values, color='lightgreen', alpha=0.7)
        ax3.set_xlabel('Group Size')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Group Size Frequency', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Group size by lifecycle stage
        stage_size = self.df.groupby('lifecycle_stage')['member_count'].mean()
        ax4.bar(stage_size.index, stage_size.values, color=['red', 'blue', 'green'], alpha=0.7)
        ax4.set_xlabel('Lifecycle Stage')
        ax4.set_ylabel('Average Group Size')
        ax4.set_title('Average Group Size by Lifecycle Stage', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Group size analysis saved: {save_path}")
    
    def create_activity_heatmap(self, save_path="output/activity_heatmap.png"):
        """Create a heatmap showing activity patterns over time"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create time bins (every 0.5 seconds)
        time_bins = np.arange(0, self.df['time_seconds'].max() + 0.5, 0.5)
        self.df['time_bin'] = pd.cut(self.df['time_seconds'], bins=time_bins, labels=False)
        
        # Count events per time bin
        activity_matrix = self.df.groupby(['time_bin', 'lifecycle_stage']).size().unstack(fill_value=0)
        
        # Create heatmap
        sns.heatmap(activity_matrix.T, cmap='YlOrRd', annot=True, fmt='d', 
                   cbar_kws={'label': 'Number of Events'}, ax=ax)
        ax.set_xlabel('Time Bin (0.5 second intervals)')
        ax.set_ylabel('Lifecycle Stage')
        ax.set_title('Activity Heatmap Over Time', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Activity heatmap saved: {save_path}")
    
    def create_member_network_analysis(self, save_path="output/member_network.png"):
        """Create a network visualization of member interactions"""
        try:
            import networkx as nx
        except ImportError:
            print("⚠️ NetworkX not installed. Skipping network visualization.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create network graph
        G = nx.Graph()
        
        # Add edges based on member interactions
        for _, row in self.df.iterrows():
            if pd.notna(row['member_ids']) and row['member_ids'] != '':
                members = str(row['member_ids']).split('-')
                # Add edges between all members in the group
                for i in range(len(members)):
                    for j in range(i+1, len(members)):
                        if G.has_edge(members[i], members[j]):
                            G[members[i]][members[j]]['weight'] += 1
                        else:
                            G.add_edge(members[i], members[j], weight=1)
        
        # Calculate node sizes based on degree
        node_sizes = [G.degree(node) * 100 for node in G.nodes()]
        
        # Draw the network
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw(G, pos, with_labels=True, node_size=node_sizes, 
               node_color='lightblue', font_size=8, font_weight='bold',
               edge_color='gray', width=1, alpha=0.7)
        
        ax.set_title('Member Interaction Network', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Member network analysis saved: {save_path}")
    
    def create_summary_dashboard(self, save_path="output/summary_dashboard.png"):
        """Create a comprehensive summary dashboard"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Key metrics summary
        ax1 = fig.add_subplot(gs[0, :2])
        total_groups = self.df['group_id'].nunique()
        total_events = len(self.df)
        avg_dwell = self.df['dwell_time_seconds'].mean()
        max_dwell = self.df['dwell_time_seconds'].max()
        
        metrics_text = f"""
        📊 KEY METRICS SUMMARY
        
        Total Groups: {total_groups}
        Total Events: {total_events}
        Average Dwell Time: {avg_dwell:.1f}s
        Maximum Dwell Time: {max_dwell:.1f}s
        Video Duration: {self.df['time_seconds'].max():.1f}s
        """
        ax1.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Summary Dashboard', fontsize=16, fontweight='bold')
        
        # Dwell time distribution
        ax2 = fig.add_subplot(gs[0, 2:])
        dwell_data = self.df[self.df['dwell_time_seconds'] > 0]
        ax2.hist(dwell_data['dwell_time_seconds'], bins=15, alpha=0.7, color='skyblue')
        ax2.set_xlabel('Dwell Time (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Dwell Time Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Group size over time
        ax3 = fig.add_subplot(gs[1, :])
        ax3.scatter(self.df['time_seconds'], self.df['member_count'], 
                   alpha=0.6, c=self.df['group_id'], cmap='tab10')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Group Size')
        ax3.set_title('Group Size Over Time', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Activity timeline
        ax4 = fig.add_subplot(gs[2, :])
        formations = self.df[self.df['is_formation'] == True]
        dispersals = self.df[self.df['is_dispersal'] == True]
        ax4.scatter(formations['time_seconds'], [1]*len(formations), 
                   c='green', s=100, alpha=0.7, label='Formations')
        ax4.scatter(dispersals['time_seconds'], [0]*len(dispersals), 
                   c='red', s=100, alpha=0.7, label='Dispersals')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Event Type')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Dispersals', 'Formations'])
        ax4.set_title('Group Lifecycle Events', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Group size distribution
        ax5 = fig.add_subplot(gs[3, :2])
        size_counts = self.df['member_count'].value_counts().sort_index()
        ax5.bar(size_counts.index, size_counts.values, color='lightgreen', alpha=0.7)
        ax5.set_xlabel('Group Size')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Group Size Distribution', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Lifecycle stage distribution
        ax6 = fig.add_subplot(gs[3, 2:])
        stage_counts = self.df['lifecycle_stage'].value_counts()
        colors = ['red', 'blue', 'green']
        ax6.pie(stage_counts.values, labels=stage_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax6.set_title('Lifecycle Stage Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Summary dashboard saved: {save_path}")
    
    def generate_all_visualizations(self):
        """Generate all visualization types"""
        print("🎨 Generating Group Behavior Visualizations...")
        print("=" * 50)
        
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        
        try:
            # Generate all visualizations
            self.create_timeline_visualization()
            self.create_dwell_time_analysis()
            self.create_group_size_analysis()
            self.create_activity_heatmap()
            self.create_member_network_analysis()
            self.create_summary_dashboard()
            
            print("\n✅ All visualizations generated successfully!")
            print("\n📁 Generated files:")
            print("   • Timeline Analysis: output/timeline_analysis.png")
            print("   • Dwell Time Analysis: output/dwell_time_analysis.png")
            print("   • Group Size Analysis: output/group_size_analysis.png")
            print("   • Activity Heatmap: output/activity_heatmap.png")
            print("   • Member Network: output/member_network.png")
            print("   • Summary Dashboard: output/summary_dashboard.png")
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")

def main():
    """Main function to run the visualization tool"""
    enhanced_csv_file = "output/enhanced_group_analysis.csv"
    
    try:
        # Initialize visualizer
        visualizer = GroupBehaviorVisualizer(enhanced_csv_file)
        
        # Generate all visualizations
        visualizer.generate_all_visualizations()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run analyze_results.py first to generate the enhanced CSV file.")
    except Exception as e:
        print(f"❌ Error during visualization: {e}")

if __name__ == "__main__":
    main() 