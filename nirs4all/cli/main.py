import click
from nirs4all.core.config import Config
from nirs4all.cli.presets import load_preset, apply_preset_to_config
import os
import json  # Added missing import

@click.group(invoke_without_command=True)
@click.option('--config', 'config_file_path',
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help='Path to a global JSON configuration file.')
@click.option('--data-path', 'data_path_override',
              type=click.Path(exists=True, resolve_path=True),
              help='Global path to the data, overrides config if both are present.')
@click.pass_context
def nirs4all_cli(ctx, config_file_path, data_path_override):
    """nirs4all: A comprehensive command-line interface for NIRS data analysis."""
    ctx.ensure_object(dict)
    
    loaded_config = None
    if config_file_path:
        try:
            loaded_config = Config.from_json_file(config_file_path)
            click.echo(f"Loaded global configuration from: {config_file_path}")
        except Exception as e:
            raise click.ClickException(f"Error loading global configuration file {config_file_path}: {e}")
    
    ctx.obj['initial_config'] = loaded_config
    ctx.obj['data_path_override'] = data_path_override

    if ctx.invoked_subcommand is None:
        if loaded_config:
            action = None
            if loaded_config.experiment and isinstance(loaded_config.experiment, dict):
                action = loaded_config.experiment.get('action')

            if action:
                click.echo(f"Action '{action}' inferred from global config.")
                if action == 'train':
                    ctx.forward(train)
                elif action == 'predict':
                    ctx.forward(predict)
                elif action == 'finetune':
                    ctx.forward(finetune)
                else:
                    click.echo(f"Unknown action '{action}' in config. Please specify a command.", err=True)
            else:
                click.echo("No command specified and no 'action' found in global config. Use --help for options.", err=True)
        elif data_path_override:
            click.echo("Only --data-path provided without a command or --config. Please specify a command.", err=True)
        else:
            click.echo(ctx.get_help())

def _get_final_config(ctx, preset_name: str = None, command_specific_data_path: str = None) -> Config:
    """Helper to construct the final config based on global options and command-specific preset/data_path."""
    initial_config_obj = ctx.obj.get('initial_config')
    global_data_path_override = ctx.obj.get('data_path_override')

    # Start with a copy of the initial config or a new one
    if initial_config_obj:
        # Create a new config from the dict representation of the loaded one to ensure it's a copy
        current_config = Config.from_dict(initial_config_obj.to_dict())
        click.echo("Starting with a copy of the loaded global config.")
    else:
        current_config = Config(dataset=None) 
        click.echo("No global config loaded, starting with a default config.")

    # 1. Apply preset first, as it might define a base dataset path or other crucial settings
    if preset_name:
        preset_data = load_preset(preset_name)
        if preset_data:
            current_config = apply_preset_to_config(current_config, preset_data)
            click.echo(f"Applied preset '{preset_name}'.")

    # 2. Apply global data_path_override (overrides preset's dataset path if any)
    if global_data_path_override:
        current_config.dataset = global_data_path_override
        click.echo(f"Applied global --data-path override: {global_data_path_override}")

    # 3. Apply command_specific_data_path (overrides global and preset's dataset path)
    if command_specific_data_path:
        current_config.dataset = command_specific_data_path
        click.echo(f"Applied command-specific --data-path: {command_specific_data_path}")
    
    if current_config.dataset:
        click.echo(f"Final effective dataset path: {current_config.dataset}")
    else:
        raise click.ClickException(
            "Dataset path is required but not found. "
            "Provide it via --data-path (global or command-specific), in a global config, or via a preset."
        )
        
    return current_config

@nirs4all_cli.command()
@click.option('--preset', 'train_preset', help='Name of the training preset to use.')
@click.pass_context
def train(ctx, train_preset):
    """Train a new model."""
    click.echo("Executing CLI command: train")
    config = _get_final_config(ctx, preset_name=train_preset)
    
    click.echo(f"Final configuration for training - Dataset: {config.dataset}")
    if config.model:
        click.echo(f"Model (potentially from preset/config): {config.model}")  # Displaying model representation
    # TODO: Actual training logic using 'config'


@nirs4all_cli.command()
@click.option('--preset', 'predict_preset', help='Name of the prediction preset to use.')
@click.option('--data-path', 'predict_data_path', 
              type=click.Path(exists=True, resolve_path=True), 
              help='Path to the new data for prediction. Overrides global --data-path for this command.')
@click.pass_context
def predict(ctx, predict_preset, predict_data_path):
    """Make predictions using a trained model."""
    click.echo("Executing CLI command: predict")
    
    config = _get_final_config(ctx, preset_name=predict_preset, command_specific_data_path=predict_data_path)
    
    click.echo(f"Final configuration for prediction - Dataset: {config.dataset}")
    if not config.model:
        click.echo("Warning: No model configured for prediction. Ensure config or preset defines a model.", err=True)
    # TODO: Actual prediction logic


@nirs4all_cli.command()
@click.option('--preset', 'finetune_preset', help='Name of the finetuning preset to use.')
@click.option('--data-path', 'finetune_data_path', 
              type=click.Path(exists=True, resolve_path=True),
              help='Path to the new data for finetuning. Overrides global --data-path for this command.')
@click.pass_context
def finetune(ctx, finetune_preset, finetune_data_path):
    """Finetune an existing model."""
    click.echo("Executing CLI command: finetune")
        
    config = _get_final_config(ctx, preset_name=finetune_preset, command_specific_data_path=finetune_data_path)

    click.echo(f"Final configuration for finetuning - Dataset: {config.dataset}")
    if not config.model:
        click.echo("Warning: No model configured for finetuning. Ensure config or preset defines the base model.", err=True)
    # TODO: Actual finetuning logic

if __name__ == '__main__':
    # Create dummy files for click.Path(exists=True) to work in a test environment
    
    # Ensure dummy files/dirs for global options
    if not os.path.exists("dummy_global_config.json"):
        with open("dummy_global_config.json", "w") as f:
            json.dump({"experiment": {"action": "train"}, "dataset": "dummy_data_dir_global"}, f)
    if not os.path.exists("dummy_data_dir_global"):
        os.makedirs("dummy_data_dir_global", exist_ok=True)
        with open(os.path.join("dummy_data_dir_global", "data.txt"), "w") as f:
            f.write("global data")

    # Ensure dummy files/dirs for command-specific options if they differ
    if not os.path.exists("dummy_predict_data_dir"):
        os.makedirs("dummy_predict_data_dir", exist_ok=True)
        with open(os.path.join("dummy_predict_data_dir", "predict_data.txt"), "w") as f:
            f.write("predict data")
        
    if not os.path.exists("dummy_finetune_data_dir"):
        os.makedirs("dummy_finetune_data_dir", exist_ok=True)
        with open(os.path.join("dummy_finetune_data_dir", "finetune_data.txt"), "w") as f:
            f.write("finetune data")

    # Ensure presets directory and a dummy preset for testing
    presets_cli_dir = os.path.join("nirs4all", "cli", "presets")
    if not os.path.exists(presets_cli_dir):
        os.makedirs(presets_cli_dir, exist_ok=True)
    dummy_preset_path = os.path.join(presets_cli_dir, "dummy_train_preset.json")
    if not os.path.exists(dummy_preset_path):
        with open(dummy_preset_path, "w") as f:
            json.dump({"model": {"class": "dummy.Model"}, "dataset": "dummy_data_dir_preset"}, f)
    if not os.path.exists("dummy_data_dir_preset"):
        os.makedirs("dummy_data_dir_preset", exist_ok=True)

    nirs4all_cli()
