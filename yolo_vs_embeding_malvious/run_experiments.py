#!/usr/bin/env python3

"""
Master Orchestration Script for Retail Detection Experiments
Handles complete pipeline:
1. Training/tuning YOLO models
2. Populating Milvus with embeddings
3. Running all experiments
4. Generating comparison reports and visualizations
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ExperimentOrchestrator:
    """Orchestrate complete experimental pipeline"""
    
    def __init__(self, config_file: str):
        """
        Initialize orchestrator
        
        Args:
            config_file: Path to experiment configuration YAML
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.results_dir = Path(self.config.get('results_dir', './results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self) -> Dict:
        """Load experiment configuration"""
        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def run_yolo_training(self, model_type: str, output_name: str) -> str:
        """
        Train or tune YOLO model (or use existing pretrained model)
        
        Args:
            model_type: yolov8 or yolov11
            output_name: Name for output model
            
        Returns:
            Path to trained model
        """
        train_config = self.config['training']
        
        # Check for existing pretrained model first
        pretrained_key = f'{model_type}_pretrained'
        if pretrained_key in train_config and train_config[pretrained_key]:
            pretrained_path = Path(train_config[pretrained_key])
            if pretrained_path.exists():
                print(f"\n{'='*80}")
                print(f"Using Existing Pretrained {model_type.upper()} Model")
                print(f"{'='*80}\n")
                print(f"✅ Found pretrained model: {pretrained_path}")
                print(f"   Skipping training for {output_name}")
                print(f"   Set 'force_retrain: true' in config to retrain anyway\n")
                return str(pretrained_path)
        
        # Check if force_retrain is disabled
        if not train_config.get('force_retrain', True):
            # Look for previously trained models in results directory
            possible_paths = [
                self.results_dir / 'training' / output_name / 'weights' / 'best.pt',
                self.results_dir / 'tuning' / output_name / 'weights' / 'best.pt',
                Path(f'runs/train/{output_name}/weights/best.pt'),
                Path(f'runs/tune/{output_name}/weights/best.pt')
            ]
            
            for model_path in possible_paths:
                if model_path.exists():
                    print(f"\n{'='*80}")
                    print(f"Using Previously Trained {model_type.upper()} Model")
                    print(f"{'='*80}\n")
                    print(f"✅ Found existing model: {model_path}")
                    print(f"   Skipping training for {output_name}")
                    print(f"   Set 'force_retrain: true' to retrain\n")
                    return str(model_path)
        
        # No existing model found or force_retrain is True - proceed with training
        print(f"\n{'='*80}")
        print(f"Training {model_type.upper()} Model: {output_name}")
        print(f"{'='*80}\n")
        
        # Check if tuning or standard training
        if train_config.get('use_tuning', False):
            cmd = [
                'python', 'yolo_tuning.py',
                '--data', train_config['dataset_yaml'],
                '--model', train_config[f'{model_type}_base'],
                '--imgsz', str(train_config['imgsz']),
                '--epochs', str(train_config['tuning_epochs']),
                '--batch', str(train_config['batch_size']),
                '--iterations', str(train_config['tuning_iterations']),
                '--device', train_config['device'],
                '--project', str(self.results_dir / 'tuning'),
                '--name', output_name,
                '--mlflow-uri', train_config['mlflow_uri'],
                '--save-checkpoints'
            ]
        else:
            # Standard training
            from ultralytics import YOLO
            
            model = YOLO(train_config[f'{model_type}_base'])
            results = model.train(
                data=train_config['dataset_yaml'],
                epochs=train_config['epochs'],
                imgsz=train_config['imgsz'],
                batch=train_config['batch_size'],
                device=train_config['device'],
                project=str(self.results_dir / 'training'),
                name=output_name,
                save=True,
                plots=True
            )
            
            return str(self.results_dir / 'training' / output_name / 'weights' / 'best.pt')
        
        # Run tuning command
        result = subprocess.run(cmd, check=True)
        
        # Find best model
        tune_dir = self.results_dir / 'tuning' / output_name / 'weights'
        best_model = tune_dir / 'best.pt'
        
        if not best_model.exists():
            best_model = tune_dir / 'last.pt'
        
        if not best_model.exists():
            raise FileNotFoundError(f"Trained model not found in {tune_dir}")
        
        print(f"\n✅ Training complete: {best_model}")
        return str(best_model)
    
    def populate_milvus(self) -> str:
        """
        Populate Milvus database with DOLG embeddings
        
        Returns:
            Path to Milvus database
        """
        print(f"\n{'='*80}")
        print("Populating Milvus Database with DOLG Embeddings")
        print(f"{'='*80}\n")
        
        milvus_config = self.config['milvus']
        
        cmd = [
            'python', 'populate_milvus_embeddings.py',
            '--dataset', self.config['training']['dataset_yaml'],
            '--dolg-model', milvus_config['dolg_model_path'],
            '--milvus-db', str(self.results_dir / milvus_config['db_name']),
            '--collection', milvus_config['collection_name'],
            '--max-templates', str(milvus_config['max_templates_per_class']),
            '--cache', str(self.results_dir / 'embedding_cache.pkl'),
            '--device', self.config['training']['device']
        ]
        
        subprocess.run(cmd, check=True)
        
        milvus_db_path = str(self.results_dir / milvus_config['db_name'])
        print(f"\n✅ Milvus database populated: {milvus_db_path}")
        return milvus_db_path
    
    def run_experiments(self, trained_models: Dict[str, str], milvus_db: str) -> Dict:
        """
        Run all experiments
        
        Args:
            trained_models: Dictionary mapping model names to paths
            milvus_db: Path to Milvus database
            
        Returns:
            Dictionary of experiment results
        """
        print(f"\n{'='*80}")
        print("Running Comparative Experiments")
        print(f"{'='*80}\n")
        
        experiment_config = self.config['experiments']
        
        # Build command
        cmd = [
            'python', 'experimental_framework.py',
            '--dataset', self.config['training']['dataset_yaml'],
            '--yolov8-model', trained_models.get('yolov8', 'yolov8m.pt'),
            '--yolov11-model', trained_models.get('yolov11', 'yolo11m.pt'),
            '--dolg-model', self.config['milvus']['dolg_model_path'],
            '--imgsz', str(self.config['training']['imgsz']),
            '--batch', str(self.config['training']['batch_size']),
            '--device', self.config['training']['device'],
            '--similarity-threshold', str(experiment_config['similarity_threshold']),
            '--mlflow-uri', self.config['training']['mlflow_uri']
        ]
        
        if experiment_config.get('run_all', True):
            cmd.append('--run-all')
        elif experiment_config.get('baseline_only', False):
            cmd.append('--run-baseline-only')
        
        subprocess.run(cmd, check=True)
        
        # Load results
        results_file = Path('experiment_comparison.json')
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results
        
        return {}
    
    def generate_visualizations(self, results: Dict):
        """
        Generate comparison visualizations
        
        Args:
            results: Dictionary of experiment results
        """
        print(f"\n{'='*80}")
        print("Generating Visualization Reports")
        print(f"{'='*80}\n")
        
        viz_dir = self.results_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Convert results to DataFrame
        df_data = []
        for exp_name, metrics in results.items():
            df_data.append({
                'Experiment': exp_name,
                'mAP@0.5': metrics['map50'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'FPS': metrics['fps'],
                'Inference Time (ms)': metrics['inference_time_ms'],
                'Milvus Hit Rate': metrics.get('milvus_hit_rate', 0) * 100 if metrics.get('milvus_hit_rate') else 0
            })
        
        df = pd.DataFrame(df_data)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        
        # 1. Detection Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detection Performance Comparison', fontsize=16, fontweight='bold')
        
        # mAP comparison
        axes[0, 0].bar(df['Experiment'], df['mAP@0.5'], color='steelblue')
        axes[0, 0].set_title('mAP@0.5 Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('mAP@0.5')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Precision/Recall comparison
        x = range(len(df))
        width = 0.35
        axes[0, 1].bar([i - width/2 for i in x], df['Precision'], width, label='Precision', color='green', alpha=0.7)
        axes[0, 1].bar([i + width/2 for i in x], df['Recall'], width, label='Recall', color='orange', alpha=0.7)
        axes[0, 1].set_title('Precision & Recall Comparison', fontweight='bold')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(df['Experiment'], rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # F1-Score comparison
        axes[1, 0].bar(df['Experiment'], df['F1-Score'], color='purple', alpha=0.7)
        axes[1, 0].set_title('F1-Score Comparison', fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Combined metrics radar chart
        from math import pi
        categories = ['mAP@0.5', 'Precision', 'Recall', 'F1-Score']
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        ax = axes[1, 1]
        ax = plt.subplot(2, 2, 4, projection='polar')
        
        for idx, row in df.iterrows():
            values = [row['mAP@0.5'], row['Precision'], row['Recall'], row['F1-Score']]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Experiment'])
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Performance Profile', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'detection_performance.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {viz_dir / 'detection_performance.png'}")
        
        # 2. Speed Performance Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Speed Performance Comparison', fontsize=16, fontweight='bold')
        
        # FPS comparison
        axes[0].bar(df['Experiment'], df['FPS'], color='teal')
        axes[0].set_title('Frames Per Second (FPS)', fontweight='bold')
        axes[0].set_ylabel('FPS')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].axhline(y=30, color='r', linestyle='--', label='Real-time threshold (30 FPS)')
        axes[0].legend()
        
        # Inference time comparison
        axes[1].bar(df['Experiment'], df['Inference Time (ms)'], color='coral')
        axes[1].set_title('Inference Time per Image', fontweight='bold')
        axes[1].set_ylabel('Time (ms)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'speed_performance.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {viz_dir / 'speed_performance.png'}")
        
        # 3. Milvus Integration Analysis (if applicable)
        milvus_experiments = df[df['Milvus Hit Rate'] > 0]
        if not milvus_experiments.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('Milvus Integration Analysis', fontsize=16, fontweight='bold')
            
            # Milvus hit rate
            axes[0].bar(milvus_experiments['Experiment'], milvus_experiments['Milvus Hit Rate'], color='gold')
            axes[0].set_title('Milvus Hit Rate', fontweight='bold')
            axes[0].set_ylabel('Hit Rate (%)')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(axis='y', alpha=0.3)
            
            # Accuracy vs Milvus usage
            axes[1].scatter(milvus_experiments['Milvus Hit Rate'], milvus_experiments['mAP@0.5'], 
                          s=200, c='purple', alpha=0.6)
            for idx, row in milvus_experiments.iterrows():
                axes[1].annotate(row['Experiment'], 
                               (row['Milvus Hit Rate'], row['mAP@0.5']),
                               fontsize=8, ha='center')
            axes[1].set_xlabel('Milvus Hit Rate (%)')
            axes[1].set_ylabel('mAP@0.5')
            axes[1].set_title('Accuracy vs Milvus Usage', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'milvus_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {viz_dir / 'milvus_analysis.png'}")
        
        # 4. Summary Table
        fig, ax = plt.subplots(figsize=(16, len(df) * 0.6 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['Experiment', 'mAP@0.5', 'Precision', 'Recall', 'F1', 'FPS', 'Time (ms)']
        
        for _, row in df.iterrows():
            table_data.append([
                row['Experiment'],
                f"{row['mAP@0.5']:.4f}",
                f"{row['Precision']:.4f}",
                f"{row['Recall']:.4f}",
                f"{row['F1-Score']:.4f}",
                f"{row['FPS']:.2f}",
                f"{row['Inference Time (ms)']:.2f}"
            ])
        
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center',
                        colWidths=[0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.12])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Experiment Comparison Summary', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(viz_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {viz_dir / 'summary_table.png'}")
        
        # Save DataFrame as CSV
        csv_path = viz_dir / 'results_summary.csv'
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path}")
        
        print(f"\n📊 All visualizations saved to: {viz_dir}")
    
    def generate_report(self, results: Dict):
        """
        Generate comprehensive markdown report
        
        Args:
            results: Dictionary of experiment results
        """
        print(f"\n{'='*80}")
        print("Generating Comprehensive Report")
        print(f"{'='*80}\n")
        
        report_path = self.results_dir / 'EXPERIMENT_REPORT.md'
        
        with open(report_path, 'w') as f:
            f.write("# Retail Item Detection - Experiment Comparison Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Find best models
            best_map = max(results.items(), key=lambda x: x[1]['map50'])
            best_speed = max(results.items(), key=lambda x: x[1]['fps'])
            best_f1 = max(results.items(), key=lambda x: x[1]['f1_score'])
            
            f.write(f"- **Best mAP@0.5**: {best_map[0]} ({best_map[1]['map50']:.4f})\n")
            f.write(f"- **Best F1-Score**: {best_f1[0]} ({best_f1[1]['f1_score']:.4f})\n")
            f.write(f"- **Fastest**: {best_speed[0]} ({best_speed[1]['fps']:.2f} FPS)\n\n")
            
            f.write("## Experiment Configuration\n\n")
            f.write(f"- **Dataset**: {self.config['training']['dataset_yaml']}\n")
            f.write(f"- **Number of Classes**: 488\n")
            f.write(f"- **Image Size**: {self.config['training']['imgsz']}px\n")
            f.write(f"- **Batch Size**: {self.config['training']['batch_size']}\n")
            f.write(f"- **Device**: {self.config['training']['device']}\n\n")
            
            f.write("## Detailed Results\n\n")
            
            for exp_name, metrics in results.items():
                f.write(f"### {exp_name}\n\n")
                f.write("**Detection Metrics:**\n")
                f.write(f"- mAP@0.5: {metrics['map50']:.4f}\n")
                f.write(f"- mAP@0.5:0.95: {metrics['map50_95']:.4f}\n")
                f.write(f"- Precision: {metrics['precision']:.4f}\n")
                f.write(f"- Recall: {metrics['recall']:.4f}\n")
                f.write(f"- F1-Score: {metrics['f1_score']:.4f}\n\n")
                
                f.write("**Speed Metrics:**\n")
                f.write(f"- Inference Time: {metrics['inference_time_ms']:.2f} ms\n")
                f.write(f"- FPS: {metrics['fps']:.2f}\n")
                f.write(f"- Preprocess Time: {metrics['preprocess_time_ms']:.2f} ms\n")
                f.write(f"- Postprocess Time: {metrics['postprocess_time_ms']:.2f} ms\n\n")
                
                if metrics.get('milvus_hit_rate'):
                    f.write("**Milvus Integration:**\n")
                    f.write(f"- Hit Rate: {metrics['milvus_hit_rate']*100:.2f}%\n")
                    f.write(f"- Embedding Time: {metrics.get('embedding_time_ms', 0):.2f} ms\n")
                    f.write(f"- Search Time: {metrics.get('similarity_search_time_ms', 0):.2f} ms\n\n")
                
                f.write("---\n\n")
            
            f.write("## Visualizations\n\n")
            f.write("See the `visualizations/` directory for:\n")
            f.write("- Detection performance comparison charts\n")
            f.write("- Speed analysis graphs\n")
            f.write("- Milvus integration analysis\n")
            f.write("- Summary tables and CSV exports\n\n")
            
            f.write("## Recommendations\n\n")
            
            # Analyze results and provide recommendations
            baseline_map = results.get('YOLOv8_Baseline_488_Classes', {}).get('map50', 0)
            hybrid_results = [v for k, v in results.items() if 'Hybrid' in k or 'Milvus' in k]
            
            if hybrid_results:
                best_hybrid = max(hybrid_results, key=lambda x: x['map50'])
                if best_hybrid['map50'] > baseline_map:
                    f.write("✅ **Hybrid approach with DOLG + Milvus shows improvement over baseline**\n\n")
                    improvement = ((best_hybrid['map50'] - baseline_map) / baseline_map) * 100
                    f.write(f"- Accuracy improvement: +{improvement:.2f}%\n")
                    f.write(f"- Recommended for production deployment\n\n")
                else:
                    f.write("⚠️ **Hybrid approach did not outperform baseline**\n\n")
                    f.write("- Consider adjusting similarity thresholds\n")
                    f.write("- May need more diverse training templates\n")
                    f.write("- Baseline YOLO may be sufficient for current use case\n\n")
            
            f.write("## MLflow Tracking\n\n")
            f.write(f"All experiments are logged to MLflow. View results:\n")
            f.write(f"```bash\n")
            f.write(f"mlflow ui --backend-store-uri {self.config['training']['mlflow_uri']}\n")
            f.write(f"```\n\n")
        
        print(f"✅ Report saved to: {report_path}")
    
    def run_complete_pipeline(self):
        """Run complete experimental pipeline"""
        print(f"\n{'#'*80}")
        print("# RETAIL ITEM DETECTION - COMPLETE EXPERIMENTAL PIPELINE")
        print(f"{'#'*80}\n")
        
        trained_models = {}
        train_config = self.config['training']
        
        # Step 1: Determine YOLOv8 model path
        if not train_config.get('train_yolov8', True):
            # Skip YOLOv8 training completely
            print(f"\n⏭️  Skipping YOLOv8 training (train_yolov8: false)")
            trained_models['yolov8'] = train_config['yolov8_base']
        else:
            # Check for pretrained or train
            yolov8_model = self.run_yolo_training('yolov8', 'yolov8_488_classes')
            trained_models['yolov8'] = yolov8_model
        
        # Step 2: Determine YOLOv11 model path
        if not train_config.get('train_yolov11', True):
            # Skip YOLOv11 training completely
            print(f"\n⏭️  Skipping YOLOv11 training (train_yolov11: false)")
            trained_models['yolov11'] = train_config['yolov11_base']
        else:
            # Check for pretrained or train
            yolov11_model = self.run_yolo_training('yolov11', 'yolov11_488_classes')
            trained_models['yolov11'] = yolov11_model
        
        print(f"\n📋 Model Summary:")
        print(f"   YOLOv8: {trained_models['yolov8']}")
        print(f"   YOLOv11: {trained_models['yolov11']}\n")
        
        # Step 3: Populate Milvus
        if self.config['experiments'].get('use_milvus', True):
            milvus_db = self.populate_milvus()
        else:
            milvus_db = None
        
        # Step 4: Run experiments
        results = self.run_experiments(trained_models, milvus_db)
        
        # Step 5: Generate visualizations
        if results:
            self.generate_visualizations(results)
            self.generate_report(results)
        
        print(f"\n{'#'*80}")
        print("# PIPELINE COMPLETE!")
        print(f"{'#'*80}\n")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"📊 View MLflow UI: mlflow ui --backend-store-uri {self.config['training']['mlflow_uri']}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Orchestrate complete retail detection experimental pipeline"
    )
    parser.add_argument("--config", type=str, default="experiment_config.yaml",
                       help="Path to experiment configuration YAML")
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        print("Creating example configuration file...")
        create_example_config(args.config)
        print(f"✅ Example configuration created: {args.config}")
        print("Please edit the configuration and run again.")
        sys.exit(1)
    
    orchestrator = ExperimentOrchestrator(args.config)
    orchestrator.run_complete_pipeline()


def create_example_config(config_path: str):
    """Create example configuration file"""
    config = {
        'results_dir': './experiment_results',
        'training': {
            'dataset_yaml': 'data/retail_488.yaml',
            'yolov8_base': 'yolov8m.pt',
            'yolov11_base': 'yolo11m.pt',
            'yolov8_pretrained': 'models/yolov8_488_classes_best.pt',  # Use this if exists
            'yolov11_pretrained': 'models/yolov11_488_classes_best.pt',  # Use this if exists
            'train_yolov8': True,
            'train_yolov11': True,
            'force_retrain': False,  # Set True to force retraining
            'use_tuning': False,
            'epochs': 100,
            'tuning_epochs': 50,
            'tuning_iterations': 30,
            'imgsz': 640,
            'batch_size': 16,
            'device': 'cuda:0',
            'mlflow_uri': 'file:./mlruns'
        },
        'milvus': {
            'dolg_model_path': 'dolg_model.pth',
            'db_name': 'milvus_retail.db',
            'collection_name': 'retail_items',
            'max_templates_per_class': 10
        },
        'experiments': {
            'run_all': True,
            'baseline_only': False,
            'use_milvus': True,
            'similarity_threshold': 0.5
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
