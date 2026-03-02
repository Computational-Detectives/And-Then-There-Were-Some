import os
import torch

from pathlib import Path
from typing import Dict


# =============== BOOKNLP FIX ===============
# A fix for BookNLP where `position_ids` are deleted, if present,
# and then reloaded. 
# 
# Source for the fix: https://github.com/booknlp/booknlp/issues/26
# ============================================
def remove_position_ids_and_save(model_file: str, device: torch.device, save_path: str):
    '''
    Loads the `state_dict` of the BERT models downloaded by BookNLP
    upon first execution. If `position_ids` exist, it deletes them,
    and stores them at the provided storage location.
    
    :param model_file: Path to the existing model file
    :param device: The `torch.device`-Object used in the pipeline 
    :param save_path: Path to the storage location of the new model file
    '''
    # Load the state dictionary
    state_dict = torch.load(model_file, map_location=device)

    # Remove the 'position_ids' key if it exists
    if "bert.embeddings.position_ids" in state_dict:
        # print(f"Removing 'position_ids' from the state dictionary of {model_file}")
        del state_dict["bert.embeddings.position_ids"]

    # Save the modified state dict to a new file
    torch.save(state_dict, save_path)
    # print(f"Modified state dict saved to {save_path}")
	
    
def process_model_files(model_params: Dict[str, str], device: torch.device) -> Dict[str, str]:
	'''
    Processes the new model files based on the provided model
	parameters.
    
    :param model_params: A dictionary of custom model parameters
    :param device: The `torch.device`-Object used in the pipeline
    '''
	updated_params = {}
	for key, path in model_params.items():
		if isinstance(path, str) and os.path.isfile(path) and path.endswith(".model"):
			save_path = path.replace(".model", "_modified.model")
			remove_position_ids_and_save(path, device, save_path)
			updated_params[key] = save_path
		else:
			updated_params[key] = path
	return updated_params


def get_model_path() -> Path:
    '''
    A convenience method to quickly retrieve the path
    at which the models are stored on the local device.
    
    :return: Returns the path to the directory of where the models are stored
    :rtype: str
    '''
    home = Path.home()
    model_path = home / 'booknlp_models'
    
    return model_path


def exists_model_path() -> bool:
    '''
    A convenience method to quickly check whether the directory
    at which the BERT models are stored already exists. Knowledge 
    of its existence is used to skip the full execution of `init_run`.
    
    :return: True, if the directory exists, else False
    :rtype: bool
    '''
    home = Path.home()
    model_path = home / "booknlp_models"
   
    if not model_path.is_dir():
        return False

    return True