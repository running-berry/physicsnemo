#!/usr/bin/env python3
# filepath: /home/master/13/dczy/running-berry/physicsnemo/dev/run_stormcast.py

import os
import sys
import subprocess
import argparse
import logging
import glob
import re
import yaml
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from ruamel.yaml.comments import CommentedSeq
import copy
from datetime import datetime
from pathlib import Path

class StormCastRunner:
    def __init__(self, config_file="master_config.yaml", log_level=logging.INFO):
        # Load configuration
        self.load_config(config_file)
        
        # Override log level if provided
        if log_level != logging.INFO:
            self.config['logging']['level'] = logging.getLevelName(log_level)
        
        # Setup logging
        self.setup_logging()
        
    def load_config(self, config_file):
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent / config_file
        
        if not config_path.exists():
            # raise error if config file does not exist
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up paths as Path objects
        self.stormcast_dir = Path(self.config['paths']['stormcast_dir'])
        self.data_dir = Path(self.config['paths']['data_dir'])
        self.rwrf_dir = Path(self.config['paths']['rwrf_dir'])
        self.log_dir = Path(self.config['paths']['log_dir'])
        
        
        # Set up experiment variables
        self.variable = self.config['experiment']['variable']
        self.experiment_name = self.config['experiment']['experiment_name']
        
        print(f"Loaded configuration from: {config_path}")
        print(f"Variable: {self.variable}")
        print(f"StormCast directory: {self.stormcast_dir}")

    def update_yaml_preserve_comments(self, target_path, updates):
        """
        Update a YAML file with values from updates dict, preserving comments.
        """
        yaml = YAML()
        yaml.preserve_quotes = True

        self.check_path_exists(target_path)
        with open(target_path, 'r') as f:
            config = yaml.load(f)

        for k, v in updates.items():
            # Force inline style for HighRes_img_size
            if isinstance(v, list):
                seq = CommentedSeq(v)
                seq.fa.set_flow_style()
                config[k] = seq
            else:
                config[k] = v
            self.logger.info(f"Updated {k} to {v} in {target_path}")

        with open(target_path, 'w') as f:
            yaml.dump(config, f)
        return target_path
    
    def sync_configs_from_master(self):
        """
        Update dataset config from master_config.yaml, preserving comments.
        """
        dataset_updates = self.config.get('dataset', {})
        print("Updating with:", dataset_updates) 
        self.update_yaml_preserve_comments(self.config['paths']['dataset_config'], dataset_updates)

        self.logger.info(f"Updated dataset config with: {dataset_updates}")
        # log dataset_config content
        self.logger.info("testing dataset config content")

        # self.logger.debug(f"Dataset config path: {self.config['paths']['dataset_config']}")
        try:
            self.logger.info(f"Reading dataset config from: {self.config['paths']['dataset_config']}")
            self.check_path_exists(self.config['paths']['dataset_config'])
            with open(self.config['paths']['dataset_config'], 'r') as f:
                dataset_config_content = f.read()
            self.logger.info(f"Dataset config content:\n{dataset_config_content}")
        except Exception as e:
            self.logger.info(f"Failed to read dataset config: {e}")

    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config['logging']['level'].upper())
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup handlers
        handlers = []
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        if self.config['logging']['console_output']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            handlers.append(console_handler)
        
        if self.config['logging']['file_output']:
            log_file_path = self.log_dir / "stormcast_runner.log"
            file_handler = logging.FileHandler(log_file_path, mode='a')
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        logging.basicConfig(level=log_level, handlers=handlers, force=True)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Logging initialized with level: {self.config['logging']['level']}")
        
    def get_experiment_name(self, model_type, experiment_name=None):
        """Generate experiment name based on configuration"""
        if experiment_name:
            return experiment_name
        
        if self.experiment_name:
            return self.experiment_name
            
        suffix_map = {
            'regression': self.config['experiment']['regression_experiment_suffix'],
            'diffusion': self.config['experiment']['diffusion_experiment_suffix'],
            'inference': self.config['experiment']['inference_experiment_suffix']
        }
        
        suffix = suffix_map.get(model_type, model_type)
        return f"{suffix}-{self.variable}"
        
    def find_latest_regression_checkpoint(self, regression_exp_name=None):
        """Find the latest regression checkpoint (.mdlus file) by numbering"""
        if regression_exp_name is None:
            regression_exp_name = self.get_experiment_name('regression')
        
        checkpoint_dir = Path(self.config['paths']['workspace_root']) / "examples/weather/stormcast/rundir" / regression_exp_name / "0" / "checkpoints_regression"
        
        if not checkpoint_dir.exists():
            self.logger.error(f"Regression checkpoint directory not found: {checkpoint_dir}")
            return None
        
        # Find all .mdlus files
        mdlus_files = list(checkpoint_dir.glob("*.mdlus"))
        
        if not mdlus_files:
            self.logger.error(f"No .mdlus files found in {checkpoint_dir}")
            return None
        
        # Extract numbers from filenames and find the latest
        latest_checkpoint = None
        latest_number = -1
        
        for mdlus_file in mdlus_files:
            match = re.search(r'\.(\d+)\.mdlus$', mdlus_file.name)
            if match:
                number = int(match.group(1))
                if number > latest_number:
                    latest_number = number
                    latest_checkpoint = mdlus_file
        
        if latest_checkpoint:
            self.logger.info(f"Found latest regression checkpoint: {latest_checkpoint}")
            return str(latest_checkpoint)
        else:
            self.logger.error("Could not find any numbered .mdlus files")
            return None

    def find_latest_diffusion_checkpoint(self, diffusion_exp_name=None):
        """Find the latest diffusion checkpoint (.mdlus file) by numbering"""
        if diffusion_exp_name is None:
            diffusion_exp_name = self.get_experiment_name('diffusion')
        
        checkpoint_dir = Path(self.config['paths']['workspace_root']) / "examples/weather/stormcast/rundir" / diffusion_exp_name / "0" / "checkpoints"
        
        if not checkpoint_dir.exists():
            self.logger.error(f"Diffusion checkpoint directory not found: {checkpoint_dir}")
            return None
        
        # Find all .mdlus files
        mdlus_files = list(checkpoint_dir.glob("*.mdlus"))
        
        if not mdlus_files:
            self.logger.error(f"No .mdlus files found in {checkpoint_dir}")
            return None
        
        # Extract numbers from filenames and find the latest
        latest_checkpoint = None
        latest_number = -1
        
        for mdlus_file in mdlus_files:
            match = re.search(r'\.(\d+)\.mdlus$', mdlus_file.name)
            if match:
                number = int(match.group(1))
                if number > latest_number:
                    latest_number = number
                    latest_checkpoint = mdlus_file
        
        if latest_checkpoint:
            self.logger.info(f"Found latest diffusion checkpoint: {latest_checkpoint}")
            return str(latest_checkpoint)
        else:
            self.logger.error("Could not find any numbered .mdlus files")
            return None
    
    def run_command(self, cmd, cwd=None, capture_output=False):
        """Run a shell command with proper logging"""
        self.logger.info(f"Executing command: {cmd}")
        if cwd:
            self.logger.info(f"Working directory: {cwd}")
        
        try:
            if capture_output:
                result = subprocess.run(cmd, shell=True, cwd=cwd, 
                                      capture_output=True, text=True, check=True)
                self.logger.debug(f"Command output: {result.stdout}")
                return result.stdout
            else:
                result = subprocess.run(cmd, shell=True, cwd=cwd, check=True)
                self.logger.info("Command completed successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed with return code {e.returncode}: {e}")
            if e.stderr:
                self.logger.error(f"Error output: {e.stderr}")
            sys.exit(1)

    def pull(self):
        """Pull latest changes from git"""
        self.logger.info("Pulling latest changes from git origin")
        self.run_command("git pull origin")
        self.logger.info("Git pull completed")

    def make_dummy(self):
        """Create dummy data"""
        self.logger.info("Creating dummy data")
        if not self.data_dir.exists():
            self.logger.warning(f"Data directory {self.data_dir} does not exist")
            return
        self.run_command("python create_dummy.py", cwd=self.data_dir)
        self.logger.info("Dummy data creation completed")

    def make_data(self):
        """Create data"""
        self.logger.info("Creating data")
        if not self.rwrf_dir.exists():
            self.logger.error(f"RWRF directory {self.rwrf_dir} does not exist")
            return
        self.run_command("python create_data.py", cwd=self.rwrf_dir)
        self.logger.info("Data creation completed")

    def ensure_log_dir(self):
        """Create log directory if it doesn't exist"""
        self.logger.info(f"Ensuring log directory exists: {self.log_dir}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Log directory ready")

    def check_required_paths(self):
        """Check if all required paths exist, raise error if any are missing."""
        required_paths = [
            self.stormcast_dir,
            self.data_dir,
            self.rwrf_dir,
            self.log_dir,
            Path(self.config['paths']['workspace_root']),
            Path(self.config['paths']['pythonpath']),
        ]
        missing = [str(p) for p in required_paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"Required path(s) do not exist: {', '.join(missing)}")
        else:
            self.logger.info("All required paths exist")

    def patch_training_config_for_diffusion(self, config_path):
        """If model_type is diffusion, patch the training config to set loss: 'edm'."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config.get('loss', None) != 'edm':
            config['loss'] = 'edm'
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
        return config_path
    
    def check_path_exists(self, path):
        """Check if a path exists, raise error if it does not."""
        if not Path(path).exists():
            self.logger.error(f"Path does not exist: {path}")
            raise FileNotFoundError(f"Path does not exist: {path}")
        self.logger.info(f"Path exists: {path}")

    def update_dataset_config(self):
        """Update dataset config to use the correct variable and experiment name."""

        config_path = self.config['paths']['dataset_config']
        self.check_path_exists(config_path)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # update variable
        config["name"] = self.config['dataset']['name']

        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
        
        return config_path

    def train(self, model_type="regression", config_name=None, experiment_name=None, profile=None):
        """Run training with logging for regression or diffusion models"""
        self.check_required_paths()
        self.sync_configs_from_master()
        self.logger.info(f"configs synced from master")
        # exit(0)  # Exit early if this is a test run

        if config_name is None:
            config_name = model_type
        
        experiment_name = self.get_experiment_name(model_type, experiment_name)
        
        self.logger.info(f"Starting {model_type} training process")
        self.logger.info(f"Experiment name: {experiment_name}")
        
        timestamp = datetime.now().strftime(self.config['logging'].get('timestamp_format', '%Y%m%d_%H%M'))
        log_file = self.log_dir / f"train_{model_type}_{experiment_name}_{timestamp}.log"
        
        # Set environment and run training
        env = os.environ.copy()
        env["PYTHONPATH"] = self.config['paths']['pythonpath']
        
        # Build torchrun command
        torchrun_cfg = self.config.get('torchrun', {})
        torchrun_args = []
        if torchrun_cfg.get('standalone', True):
            torchrun_args.append('--standalone')
        torchrun_args.append(f"--nnodes={torchrun_cfg.get('nnodes', 1)}")
        torchrun_args.append(f"--nproc_per_node={torchrun_cfg.get('nproc_per_node', 1)}")
        cmd = f"torchrun {' '.join(torchrun_args)} train.py --config-name {config_name}"
        
        # Track which parameters we've already added to avoid duplicates
        added_params = set()
        
        # Add experiment name first (most important)
        cmd += f" training.experiment_name={experiment_name}"
        added_params.add('experiment_name')
        

        # For diffusion training, automatically set regression checkpoint path
        if model_type == "diffusion":
            regression_checkpoint = self.find_latest_regression_checkpoint()
            if regression_checkpoint:
                cmd += f" model.regression_weights={regression_checkpoint}"
                self.logger.info(f"Using regression checkpoint: {regression_checkpoint}")
            else:
                self.logger.error("Cannot find regression checkpoint for diffusion training")
                return
            
            # configure diffusion-specific parameters

        # Apply training overrides
        def apply_training_overrides(model_type):
            """Apply model-specific training overrides to the training config."""
            overrides = self.config['training'].get(model_type, {})
            training_config_path = self.config['paths']['training_config']
            self.logger.info(f"Applying training overrides for {model_type}")
            self.check_path_exists(training_config_path)
            yaml = YAML()
            yaml.preserve_quotes = True
            with open(training_config_path, 'r') as f:
                training_config = yaml.load(f)
            for k, v in overrides.items():
                training_config[k] = v
                self.logger.info(f"Applied override: {k} = {v} for {model_type} training")
            with open(training_config_path, 'w') as f:
                yaml.dump(training_config, f)
                self.logger.info(f"Updated training config with overrides for {model_type}")
                
        if model_type == "diffusion":
            apply_training_overrides("diffusion")
        elif model_type == "regression":
            apply_training_overrides("regression")

        def update_model_name(model_type):
            """Update model_name in model config file based on model_type."""
            model_config_path = self.config['paths']['model_config']
            self.check_path_exists(model_config_path)
            yaml = YAML()
            yaml.preserve_quotes = True
            with open(model_config_path, 'r') as f:
                model_config = yaml.load(f)
            model_config['model_name'] = model_type
            with open(model_config_path, 'w') as f:
                yaml.dump(model_config, f)
            self.logger.info(f"Set model_name to '{model_type}' in {model_config_path}")

        update_model_name(model_type)


        # Finalize command
        cmd += f" 2>&1 | tee {log_file}"
        
        self.logger.info(f"Training command: {cmd}")
        self.logger.info("Starting training run...")
        
        try:
            result = subprocess.run(cmd, shell=True, cwd=self.stormcast_dir, env=env, check=True)
            self.logger.info(f"{model_type} training completed successfully")
            self.logger.info(f"Model saved to: rundir/{experiment_name}/0/")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"{model_type} training failed with return code {e.returncode}")
            self.logger.error("Check CUDA installation and GPU availability")
            raise

    def inference(self, config_name="stormcast", experiment_name=None):
        """Run inference with automatic checkpoint detection"""
        self.check_required_paths()
        
        experiment_name = self.get_experiment_name('inference', experiment_name)
        
        self.logger.info(f"Starting inference process")
        self.logger.info(f"Experiment name: {experiment_name}")
        
        timestamp = datetime.now().strftime(self.config['logging'].get('timestamp_format', '%Y%m%d_%H%M'))
        log_file = self.log_dir / f"inference_{experiment_name}_{timestamp}.log"
        
        # Find latest checkpoints
        regression_checkpoint = (self.config['inference'].get('regression_checkpoint') or 
                                self.find_latest_regression_checkpoint())
        diffusion_checkpoint = (self.config['inference'].get('diffusion_checkpoint') or 
                            self.find_latest_diffusion_checkpoint())
        
        if not regression_checkpoint or not diffusion_checkpoint:
            self.logger.error("Cannot find required checkpoints for inference")
            return
        
        # Set environment
        env = os.environ.copy()
        env["PYTHONPATH"] = self.config['paths']['pythonpath']
        
        # Build inference command
        torchrun_cfg = self.config.get('torchrun', {})
        torchrun_args = []
        if torchrun_cfg.get('standalone', True):
            torchrun_args.append('--standalone')
        torchrun_args.append(f"--nnodes={torchrun_cfg.get('nnodes', 1)}")
        torchrun_args.append(f"--nproc_per_node={torchrun_cfg.get('nproc_per_node', 1)}")
        
        cmd = (f"torchrun {' '.join(torchrun_args)} inference.py "
            f"--config-path config/inference --config-name {config_name} "
            f"inference.experiment_name={experiment_name} "
            f"inference.regression_checkpoint={regression_checkpoint} "
            f"inference.diffusion_checkpoint={diffusion_checkpoint}")
        
        # Add inference config overrides
        if 'inference' in self.config:
            for key, value in self.config['inference'].items():
                if key not in ['regression_checkpoint', 'diffusion_checkpoint', 'experiment_name']:
                    cmd += f" inference.{key}={value}"
        
        cmd += f" 2>&1 | tee {log_file}"
        
        self.logger.info(f"Inference command: {cmd}")
        self.logger.info("Starting inference run...")
        
        try:
            subprocess.run(cmd, shell=True, cwd=self.stormcast_dir, env=env, check=True)
            self.logger.info(f"Inference completed successfully")
            self.logger.info(f"Results saved to: rundir/{experiment_name}/0/")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Inference failed with return code {e.returncode}")
            sys.exit(1)

    def clear_data(self):
        """Remove data directories"""
        self.logger.info("Clearing data directories")
        dirs_to_remove = [
            self.data_dir / "DummyHighRes",
            self.data_dir / "DummyLowRes",
            self.data_dir / "HighRes", 
            self.data_dir / "LowRes"
        ]
        
        removed_count = 0
        for dir_path in dirs_to_remove:
            if dir_path.exists():
                self.logger.info(f"Removing directory: {dir_path}")
                try:
                    subprocess.run(f"rm -rf {dir_path}", shell=True, check=True)
                    removed_count += 1
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Failed to remove {dir_path}: {e}")
            else:
                self.logger.debug(f"Directory does not exist: {dir_path}")
        
        self.logger.info(f"Data cleanup completed. Removed {removed_count} directories")

    def clear_rundir(self):
        """Remove rundir"""
        self.check_required_paths()
        
        rundir = self.stormcast_dir / "rundir"
        if rundir.exists():
            self.logger.info(f"Removing run directory: {rundir}")
            try:
                subprocess.run(f"rm -rf {rundir}", shell=True, check=True)
                self.logger.info("Run directory cleared successfully")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to clear run directory: {e}")
        else:
            self.logger.info("Run directory does not exist, nothing to clear")

    def make_cache(self):
        """Create cache for specified variable"""
        self.logger.info(f"Creating cache for variable: {self.variable}")
        if not self.rwrf_dir.exists():
            self.logger.error(f"RWRF directory {self.rwrf_dir} does not exist")
            return
            
        cmd = f"make nc-to-npz-{self.variable}"
        self.run_command(cmd, cwd=self.rwrf_dir)
        self.logger.info(f"Cache creation completed for variable: {self.variable}")

    def run_full_pipeline(self, model_type="regression", experiment_name=None):
        """Run the complete pipeline for specified model type"""
        self.check_required_paths()
        
        exp_name = experiment_name or self.experiment_name or f"{model_type}-{self.variable}"
        
        self.logger.info("=" * 60)
        self.logger.info(f"Starting full StormCast pipeline for {model_type} model")
        self.logger.info(f"Variable: {self.variable}")
        self.logger.info(f"Experiment name: {exp_name}")
        self.logger.info("=" * 60)
        
        pipeline_steps = [
            ("Ensuring log directory", self.ensure_log_dir),
            ("Creating cache", self.make_cache),
            ("Clearing data", self.clear_data),
            ("Clearing run directory", self.clear_rundir),
            ("Making data", self.make_data),
        ]
        
        # Add appropriate training step based on model type
        if model_type == "regression":
            pipeline_steps.append(("Training regression model", 
                                 lambda: self.train(model_type="regression", experiment_name=exp_name)))
        elif model_type == "diffusion":
            pipeline_steps.append(("Training diffusion model", 
                                 lambda: self.train(model_type="diffusion", experiment_name=exp_name)))
        elif model_type == "both":
            reg_exp = f"regression-{self.variable}"
            diff_exp = f"diffusion-{self.variable}"
            pipeline_steps.append(("Training regression model", 
                                 lambda: self.train(model_type="regression", experiment_name=reg_exp)))
            pipeline_steps.append(("Training diffusion model", 
                                 lambda: self.train(model_type="diffusion", experiment_name=diff_exp)))
        
        for step_name, step_func in pipeline_steps:
            self.logger.info(f"Pipeline step: {step_name}")
            try:
                step_func()
                self.logger.info(f"✓ {step_name} completed successfully")
            except Exception as e:
                self.logger.error(f"✗ {step_name} failed: {e}")
                self.logger.error("Pipeline execution stopped due to error")
                raise
        
        self.logger.info("=" * 60)
        self.logger.info(f"Full {model_type} pipeline completed successfully!")
        self.logger.info(f"Experiment: {exp_name}")
        self.logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="StormCast training pipeline")
    parser.add_argument("command", choices=[
        "pull", "make_dummy", "make_data", "log_dir", "train", "train_regression", 
        "train_diffusion", "inference", "clear_data", "clear_rundir", "make_cache", "run"
    ], help="Command to execute")
    parser.add_argument("--config", default="master_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--variable", 
                       help="Variable to process (overrides config)")
    parser.add_argument("--model-type", default="regression",
                       choices=["regression", "diffusion", "both"],
                       help="Model type to train (default: regression)")
    parser.add_argument("--config-name", 
                       help="Custom config name to use")
    parser.add_argument("--experiment-name",
                       help="Experiment name for organizing outputs")
    parser.add_argument("--training-profile",
                       help="Training profile to use (quick_test, production, debug)")
    parser.add_argument("--log-level", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (overrides config)")
    
    args = parser.parse_args()
    
    # Convert log level string to logging constant
    log_level = getattr(logging, args.log_level.upper()) if args.log_level else logging.INFO
    
    # Initialize runner with config
    runner = StormCastRunner(config_file=args.config, log_level=log_level)
    
    # Override config with command line arguments
    if args.variable:
        runner.variable = args.variable
        runner.config['experiment']['variable'] = args.variable
    if args.experiment_name:
        runner.experiment_name = args.experiment_name
        runner.config['experiment']['experiment_name'] = args.experiment_name
    
    # Execute commands
    if args.command == "inference":
        runner.inference(config_name=args.config_name or "stormcast",
                        experiment_name=args.experiment_name)
    elif args.command == "pull":
        runner.pull()
    elif args.command == "make_dummy":
        runner.make_dummy()
    elif args.command == "make_data":
        runner.make_data()
    elif args.command == "log_dir":
        runner.ensure_log_dir()
    elif args.command == "train":
        runner.train(model_type=args.model_type, config_name=args.config_name, 
                    experiment_name=args.experiment_name, profile=args.training_profile)
    elif args.command == "train_regression":
        runner.train(model_type="regression", config_name=args.config_name or "regression",
                    experiment_name=args.experiment_name, profile=args.training_profile)
    elif args.command == "train_diffusion":
        runner.train(model_type="diffusion", config_name=args.config_name or "diffusion",
                    experiment_name=args.experiment_name, profile=args.training_profile)
    elif args.command == "clear_data":
        runner.clear_data()
    elif args.command == "clear_rundir":
        runner.clear_rundir()
    elif args.command == "make_cache":
        runner.make_cache()
    elif args.command == "run":
        runner.run_full_pipeline(model_type=args.model_type, 
                                experiment_name=args.experiment_name)
    else:
        runner.logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main()