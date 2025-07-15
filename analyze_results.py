#!/usr/bin/env python3
"""
Group Behavior Analysis Tool
Processes CSV files from the Group Detection System and generates insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class GroupBehaviorAnalyzer:
    def __init__(self, csv_file_path):
        """Initialize analyzer with CSV file path"""
        self.csv_file_path = csv_file_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the CSV data"""
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
        
        self.df = pd.read_csv(self.csv_file_path)
        
        # Convert frame numbers to seconds (assuming 30 fps)
        self.df['time_seconds'] = self.df['frame'] / 30
        
        # Identify group formations (dwell_time_frames == 0)
        self.df['is_formation'] = self.df['dwell_time_frames'] == 0
        
        # Identify group dispersals (saved_frame_path == "disappeared")
        self.df['is_dispersal'] = self.df['saved_frame_path'] == "disappeared"
        
        print(f"✅ Loaded {len(self.df)} records from {self.csv_file_path}")
    
    def generate_basic_stats(self):
        """Generate basic statistics about group behavior"""
        stats = {
            'total_groups': self.df['group_id'].nunique(),
            'total_events': len(self.df),
            'avg_group_size': self.df['member_count'].mean(),
            'max_group_size': self.df['member_count'].max(),
            'min_group_size': self.df['member_count'].min(),
            'avg_dwell_time_seconds': self.df['dwell_time_frames'].mean() / 30,
            'max_dwell_time_seconds': self.df['dwell_time_frames'].max() / 30,
            'total_duration_seconds': self.df['time_seconds'].max(),
            'group_formations': self.df['is_formation'].sum(),
            'group_dispersals': self.df['is_dispersal'].sum()
        }
        
        return stats
    
    def analyze_group_sizes(self):
        """Analyze distribution of group sizes"""
        size_distribution = self.df['member_count'].value_counts().sort_index()
        
        # Categorize group sizes
        size_categories = {
            'Small (3-4 people)': len(self.df[self.df['member_count'].between(3, 4)]),
            'Medium (5-6 people)': len(self.df[self.df['member_count'].between(5, 6)]),
            'Large (7+ people)': len(self.df[self.df['member_count'] >= 7])
        }
        
        return {
            'distribution': size_distribution,
            'categories': size_categories
        }
    
    def analyze_dwell_times(self):
        """Analyze group dwell time patterns"""
        # Convert dwell times to seconds
        dwell_times_seconds = self.df['dwell_time_frames'] / 30
        
        # Categorize dwell times
        dwell_categories = {
            'Brief (1-5s)': len(dwell_times_seconds[dwell_times_seconds.between(1, 5)]),
            'Short (5-15s)': len(dwell_times_seconds[dwell_times_seconds.between(5, 15)]),
            'Medium (15-30s)': len(dwell_times_seconds[dwell_times_seconds.between(15, 30)]),
            'Long (30s+)': len(dwell_times_seconds[dwell_times_seconds >= 30])
        }
        
        return {
            'mean_dwell_time': dwell_times_seconds.mean(),
            'median_dwell_time': dwell_times_seconds.median(),
            'std_dwell_time': dwell_times_seconds.std(),
            'categories': dwell_categories,
            'all_dwell_times': dwell_times_seconds
        }
    
    def analyze_temporal_patterns(self):
        """Analyze group formation patterns over time"""
        formations = self.df[self.df['is_formation'] == True].copy()
        
        if len(formations) == 0:
            return {"error": "No group formations found"}
        
        # Time-based analysis
        formations['minute'] = formations['time_seconds'] // 60
        
        temporal_stats = {
            'formations_per_minute': formations.groupby('minute').size(),
            'peak_formation_time': formations['time_seconds'].mode().iloc[0] if len(formations['time_seconds'].mode()) > 0 else 0,
            'total_formation_time_span': formations['time_seconds'].max() - formations['time_seconds'].min(),
            'avg_time_between_formations': formations['time_seconds'].diff().mean()
        }
        
        return temporal_stats
    
    def analyze_member_patterns(self):
        """Analyze how individuals move between groups"""
        member_history = {}
        
        for _, row in self.df.iterrows():
            if pd.isna(row['member_ids']) or row['member_ids'] == '':
                continue
                
            members = str(row['member_ids']).split('-')
            group_id = row['group_id']
            
            for member in members:
                if member not in member_history:
                    member_history[member] = []
                member_history[member].append(group_id)
        
        # Analyze member behavior
        member_stats = {}
        for member, groups in member_history.items():
            member_stats[member] = {
                'total_groups': len(set(groups)),
                'group_sequence': groups,
                'is_transient': len(set(groups)) > 1,
                'is_stable': len(set(groups)) == 1
            }
        
        return member_stats
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("=" * 60)
        print("📊 GROUP BEHAVIOR ANALYSIS REPORT")
        print("=" * 60)
        
        # Basic Statistics
        stats = self.generate_basic_stats()
        print(f"\n🎯 KEY METRICS:")
        print(f"   • Total Groups Detected: {stats['total_groups']}")
        print(f"   • Total Events Logged: {stats['total_events']}")
        print(f"   • Average Group Size: {stats['avg_group_size']:.1f} people")
        print(f"   • Average Dwell Time: {stats['avg_dwell_time_seconds']:.1f} seconds")
        print(f"   • Maximum Dwell Time: {stats['max_dwell_time_seconds']:.1f} seconds")
        print(f"   • Total Video Duration: {stats['total_duration_seconds']:.1f} seconds")
        
        # Group Size Analysis
        size_analysis = self.analyze_group_sizes()
        print(f"\n👥 GROUP SIZE DISTRIBUTION:")
        for category, count in size_analysis['categories'].items():
            percentage = (count / stats['total_events']) * 100
            print(f"   • {category}: {count} events ({percentage:.1f}%)")
        
        # Dwell Time Analysis
        dwell_analysis = self.analyze_dwell_times()
        print(f"\n⏱️ DWELL TIME ANALYSIS:")
        for category, count in dwell_analysis['categories'].items():
            percentage = (count / stats['total_events']) * 100
            print(f"   • {category}: {count} events ({percentage:.1f}%)")
        
        # Temporal Patterns
        temporal_analysis = self.analyze_temporal_patterns()
        if 'error' not in temporal_analysis:
            print(f"\n📈 TEMPORAL PATTERNS:")
            print(f"   • Peak Formation Time: {temporal_analysis['peak_formation_time']:.1f} seconds")
            print(f"   • Average Time Between Formations: {temporal_analysis['avg_time_between_formations']:.1f} seconds")
        
        # Member Analysis
        member_analysis = self.analyze_member_patterns()
        stable_members = sum(1 for member_data in member_analysis.values() if member_data['is_stable'])
        transient_members = sum(1 for member_data in member_analysis.values() if member_data['is_transient'])
        
        print(f"\n👤 MEMBER BEHAVIOR:")
        print(f"   • Total Unique Members: {len(member_analysis)}")
        print(f"   • Stable Members (1 group): {stable_members}")
        print(f"   • Transient Members (multiple groups): {transient_members}")
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 60)
        
        return {
            'basic_stats': stats,
            'size_analysis': size_analysis,
            'dwell_analysis': dwell_analysis,
            'temporal_analysis': temporal_analysis,
            'member_analysis': member_analysis
        }
    
    def save_detailed_csv(self, output_path):
        """Save enhanced CSV with additional analysis columns"""
        enhanced_df = self.df.copy()
        
        # Add analysis columns
        enhanced_df['time_seconds'] = enhanced_df['frame'] / 30
        enhanced_df['dwell_time_seconds'] = enhanced_df['dwell_time_frames'] / 30
        enhanced_df['is_formation'] = enhanced_df['dwell_time_frames'] == 0
        enhanced_df['is_dispersal'] = enhanced_df['saved_frame_path'] == "disappeared"
        
        # Add group lifecycle stage
        enhanced_df['lifecycle_stage'] = 'persistence'
        enhanced_df.loc[enhanced_df['is_formation'], 'lifecycle_stage'] = 'formation'
        enhanced_df.loc[enhanced_df['is_dispersal'], 'lifecycle_stage'] = 'dispersal'
        
        # Save enhanced CSV
        enhanced_df.to_csv(output_path, index=False)
        print(f"✅ Enhanced CSV saved to: {output_path}")
        
        return enhanced_df

def main():
    """Main function to run the analysis"""
    csv_file = "output/group_analysis_log.csv"
    
    try:
        # Initialize analyzer
        analyzer = GroupBehaviorAnalyzer(csv_file)
        
        # Generate comprehensive report
        report = analyzer.generate_report()
        
        # Save enhanced CSV
        enhanced_csv_path = "output/enhanced_group_analysis.csv"
        analyzer.save_detailed_csv(enhanced_csv_path)
        
        print(f"\n📁 Files generated:")
        print(f"   • Original CSV: {csv_file}")
        print(f"   • Enhanced CSV: {enhanced_csv_path}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run the video analysis first to generate the CSV file.")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    main() 