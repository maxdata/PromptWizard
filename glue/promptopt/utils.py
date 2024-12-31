from ..exceptions import GlueValidaionException
from .constants import SupportedPromptOpt
from .core_logic import CritiqueNRefine
from .ine.base_classes import CritiqueNRefineParams, \
    CritiqueNRefinePromptPool


def get_promptopt_class(prompt_technique_name: str):
    """
    :params prompt_technique_name: Name of prompt optimization technique
    :return: Instance of class PromptRefinements, which is super class for all Prompt Optimization classes,
             Instance of class that holds all hyperparameters for that technique,
             Instance of class that holds all prompt strings for that techniques
    """
    prompt_technique_name = prompt_technique_name.lower()
    if prompt_technique_name == SupportedPromptOpt.CRITIQUE_N_REFINE.value:
        return CritiqueNRefine, CritiqueNRefineParams, CritiqueNRefinePromptPool
    else:
        raise GlueValidaionException(f"Value provided for `prompt_technique_name` field in config yaml of "
                                     f"prompt manager is `{prompt_technique_name}`, which is not a valid name for "
                                     f"the prompt optimization techniques that we support. Please provide input as one "
                                     f"among the following:  {SupportedPromptOpt.all_values()}", None)


