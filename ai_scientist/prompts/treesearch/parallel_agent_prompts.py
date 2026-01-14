RESPONSE_FORMAT = (
    "Your response should be a brief outline/sketch of your proposed solution in natural language (7-10 sentences), "
    "followed by a single markdown code block (using the format ```python ... ```) which implements this solution and prints out the evaluation metric(s) if applicable. "
    "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
    "Make sure to write concise code."
)

METRICPARSE_RESPONSE_FORMAT = (
    "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
    "followed by a single markdown code block (using the format ```python ... ```) which implements the full code for the metric parsing. "
    "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
    "Your generated code should be complete and executable. "
)

DEBUG_RESPONSE_FORMAT = (
    "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
    "followed by a single markdown code block (using the format ```python ... ```) which implements the full code including the bugfix/solution. "
    "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
    "Your generated code should be complete and executable. Do not omit any part of the code, even if it was part of a previous implementation."
    "Make sure to write concise code."
)

HYPERPARAM_CODE_RESPONSE_FORMAT = (
    "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
    "followed by a single markdown code block (using the format ```python ... ```) which implements the full code including hyperparameter tuning. "
    "There should be no additional headings or text in your response. Do not omit any part of the code, "
    "Your generated code should be complete and executable."
    "Make sure to write concise code."
)

ABLATION_RESPONSE_FORMAT = (
    "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
    "followed by a single markdown code block (using the format ```python ... ```) which implements the full code including the ablation study. "
    "There should be no additional headings or text in your response. Do not omit any part of the code, "
    "Your generated code should be complete and executable."
    "Make sure to write concise code."
)

ENVIRONMENT_PROMPT_TEMPLATE = (
    "Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
)

IMPL_GUIDELINE_BASE = [
    "CRITICAL GPU REQUIREMENTS - Your code MUST include ALL of these:",
    "  - At the start of your code, add these lines to handle GPU/CPU:",
    "    ```python",
    "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    "    print(f'Using device: {device}')",
    "    ```",
    "  - ALWAYS move models to device using the `.to(device)` method",
    "  - ALWAYS move input tensors to device using the `.to(device)` method",
    "  - ALWAYS move model related tensors to device using the `.to(device)` method",
    "  - For optimizers, create them AFTER moving model to device",
    "  - When using DataLoader, move batch tensors to device in training loop: `batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}`",
    "CRITICAL MODEL INPUT GUIDELINES:",
    "  - Always pay extra attention to the input to the model being properly normalized",
    "  - This is extremely important because the input to the model's forward pass directly affects the output, and the loss function is computed based on the output",
]

IMPL_GUIDELINE_SUFFIX = [
    "For generative modeling tasks, you must:",
    "  - Generate a set of samples from your model",
    "  - Compare these samples with ground truth data using appropriate visualizations",
    "  - When saving plots, always use the 'working_dir' variable that will be defined at the start of the script",
    "  - Make sure to give each figure a unique and appropriate name based on the dataset it represents, rather than reusing the same filename.",
    "Important code structure requirements:",
    "  - Do NOT put any execution code inside 'if __name__ == \"__main__\":' block",
    "  - All code should be at the global scope or in functions that are called from the global scope",
    "  - The script should execute immediately when run, without requiring any special entry point",
    "The code should start with:",
    "  import os",
    "  working_dir = os.path.join(os.getcwd(), 'working')",
    "  os.makedirs(working_dir, exist_ok=True)",
    "The code should be a single-file python program that is self-contained and can be executed as-is.",
    "No parts of the code should be skipped, don't terminate the code execution before finishing the script.",
    "Your response should only contain a single code block.",
    "Be aware of the running time of the code, it should complete within __EXEC_TIMEOUT__.",
    "You can also use the \"./working\" directory to store any temporary files that your code needs to create.",
    "Data saving requirements:",
    "- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()",
    "- Use the following naming convention for saved files:",
    "  ```python",
    "  # At the start of your code",
    "  experiment_data = {",
    "      'dataset_name_1': {",
    "          'metrics': {'train': [], 'val': []},",
    "          'losses': {'train': [], 'val': []},",
    "          'predictions': [],",
    "          'ground_truth': [],",
    "          # Add other relevant data",
    "      },",
    "      # Add additional datasets as needed:",
    "      'dataset_name_2': {",
    "          'metrics': {'train': [], 'val': []},",
    "          'losses': {'train': [], 'val': []},",
    "          'predictions': [],",
    "          'ground_truth': [],",
    "          # Add other relevant data",
    "      },",
    "  }",
    "  # During training/evaluation:",
    "  experiment_data['dataset_name_1']['metrics']['train'].append(train_metric)",
    "  ```",
    "- Include timestamps or epochs with the saved metrics",
    "- For large datasets, consider saving in chunks or using np.savez_compressed()",
    "CRITICAL EVALUATION REQUIREMENTS - Your code MUST include ALL of these:",
    "  1. Track and print validation loss at each epoch or at suitable intervals:",
    "     ```python",
    "     print(f'Epoch {epoch}: validation_loss = {val_loss:.4f}')",
    "     ```",
    "  2. Track and update ALL these additional metrics: __EVALUATION_METRICS__",
    "  3. Update metrics at EACH epoch:",
    "  4. Save ALL metrics at the end:",
    "     ```python",
    "     np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)",
    "     ```",
]

IMPL_GUIDELINE_KFOLD_TEMPLATE = (
    "The evaluation should be based on {k_fold}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
)

IMPL_GUIDELINE_MULTI_DATASET = [
    "You MUST evaluate your solution on at least __NUM_SYN_DATASETS__ different synthetic datasets to ensure robustness:",
    "  - Use standard benchmark datasets when available",
    "  - If using synthetic data, generate at least __NUM_SYN_DATASETS__ variants with different characteristics",
    "  - Report metrics separately for each dataset",
    "  - Compute and report the average metric across all datasets",
]

DRAFT_INTRO = (
    "You are an AI researcher who is looking to publish a paper that will contribute significantly to the field."
    "Your first task is to write a python code to implement a solid baseline based on your research idea provided below, "
    "from data preparation to model training, as well as evaluation and visualization. "
    "Focus on getting a simple but working implementation first, before any sophisticated improvements. "
    "We will explore more advanced variations in later stages."
)

DRAFT_EXPERIMENT_GUIDELINE = [
    "This first experiment design should be relatively simple, without extensive hyper-parameter optimization.",
    "Take the Memory section into consideration when proposing the design. ",
    "The solution sketch should be 6-10 sentences. ",
    "Don't suggest to do EDA.",
    "Make sure to create synthetic data if needed.",
    "",
]

DEBUG_INTRO = (
    "You are an experienced AI researcher. Your previous code for research experiment had a bug, so based on the information below, you should revise it in order to fix this bug. "
    "Your response should be an implementation outline in natural language,"
    " followed by a single markdown code block which implements the bugfix/solution."
)

DEBUG_GUIDELINE = [
    "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
    "Don't suggest to do EDA.",
]

IMPROVE_INTRO = (
    "You are an experienced AI researcher. You are provided with a previously developed "
    "implementation. Your task is to improve it based on the current experimental stage."
)

HYPERPARAM_INTRO_TEMPLATE = (
    "You are an experienced AI researcher. You are provided with a previously developed "
    "baseline implementation. Your task is to implement hyperparameter tuning for the following idea: "
    "{name}. "
    "{description}"
)

HYPERPARAM_GUIDELINE = [
    "The code should be a single-file python program that is self-contained and can be executed as-is.",
    "No parts of the code should be skipped, don't terminate the code execution before finishing the script.",
    "Data saving requirements:",
    "- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()",
    "- Use the following naming convention for saved files:",
    "  ```python",
    "  # At the start of your code",
    "  experiment_data = {",
    "      'hyperparam_tuning_type_1': {",
    "          'dataset_name_1': {",
    "              'metrics': {'train': [], 'val': []},",
    "              'losses': {'train': [], 'val': []},",
    "              'predictions': [],",
    "              'ground_truth': [],",
    "              # Add other relevant data",
    "          },",
    "          # Add additional datasets as needed:",
    "      },",
    "      # Add additional hyperparam tuning types as needed",
    "  }",
    "Make sure to use a filename 'experiment_data.npy' to save the data. Do not use any other filename.",
]

ABLATION_INTRO_TEMPLATE = (
    "You are an experienced AI researcher. You are provided with a previously developed "
    "baseline implementation. Your task is to implement the ablation study for the following idea: "
    "{name}. "
    "{description}"
)

ABLATION_GUIDELINE = [
    "The code should be a single-file python program that is self-contained and can be executed as-is.",
    "No parts of the code should be skipped, don't terminate the code execution before finishing the script.",
    "Data saving requirements:",
    "- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()",
    "- Use the following naming convention for saved files:",
    "  ```python",
    "  # At the start of your code",
    "  experiment_data = {",
    "      'ablation_type_1': {",
    "          'dataset_name_1': {",
    "              'metrics': {'train': [], 'val': []},",
    "              'losses': {'train': [], 'val': []},",
    "              'predictions': [],",
    "              'ground_truth': [],",
    "              # Add other relevant data",
    "          },",
    "          # Add additional datasets as needed:",
    "          'dataset_name_2': {",
    "              'metrics': {'train': [], 'val': []},",
    "              'losses': {'train': [], 'val': []},",
    "              'predictions': [],",
    "              'ground_truth': [],",
    "              # Add other relevant data",
    "          },",
    "      },",
    "      # Add additional ablation types as needed",
    "  }",
    "Make sure to use a filename 'experiment_data.npy' to save the data. Do not use any other filename.",
]

PLOTTING_GUIDELINE_BASE = [
    "AVAILABLE DATA: ",
    "Experiment Data: experiment_data.npy",
    "REQUIREMENTS: ",
    "The code should start with:",
    "  import matplotlib.pyplot as plt",
    "  import numpy as np",
    "  import os",
    "  working_dir = os.path.join(os.getcwd(), 'working')",
    "Create standard visualizations of experiment results",
    "Save all plots to working_dir",
    "Include training/validation curves if available",
    "ONLY plot data that exists in experiment_data.npy - DO NOT make up or simulate any values",
    "Use basic matplotlib without custom styles",
    "Each plot should be in a separate try-except block",
    "Always close figures after saving",
    "Always include a title for each plot, and be sure to use clear subtitles—such as 'Left: Ground Truth, Right: Generated Samples'—while also specifying the type of dataset being used.",
    "Make sure to use descriptive names for figures when saving e.g. always include the dataset name and the type of plot in the name",
    "When there are many similar figures to plot (e.g. generated samples at each epoch), make sure to plot only at a suitable interval of epochs so that you only plot at most 5 figures.",
    "Use the following experiment code to infer the data to plot: __EXPERIMENT_CODE__",
    "Example to extract data from experiment_data: experiment_data['dataset_name_1']['metrics']['train']",
    "Example data loading and plot saving code: ",
    """
                try:
                    experiment_data = np.load(os.path.join(working_dir, 'experiment_data.npy'), allow_pickle=True).item()
                except Exception as e:
                    print(f'Error loading experiment data: {e}')

                try:
                    # First plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_1].png')
                    plt.close()
                except Exception as e:
                    print(f\"Error creating plot1: {e}\")
                    plt.close()  # Always close figure even if error occurs

                try:
                    # Second plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_2].png')
                    plt.close()
                except Exception as e:
                    print(f\"Error creating plot2: {e}\")
                    plt.close()
            """,
]

PLOTTING_GUIDELINE_STAGE3 = [
    "IMPORTANT: Use the following base plotting code as a starting point:",
    "Base plotting code: {base_plotting_code}",
    "Modify the base plotting code to:",
    "1. Keep the same numpy data structure and plotting style",
    "2. Add comparison plots between different datasets",
    "3. Add dataset-specific visualizations if needed",
    "4. Include clear labels indicating which plots are from which dataset",
    "5. Use consistent naming conventions for saved files",
]

PLOTTING_GUIDELINE_STAGE4 = [
    "IMPORTANT: This is an ablation study. Use the following base plotting code as a starting point:",
    "Base plotting code: \n{base_plotting_code}",
    "Modify the base plotting code to:",
    "1. Keep the same numpy data structure and plotting style",
    "2. Add comparison plots between ablation and baseline results",
    "3. Add ablation-specific visualizations if needed",
    "4. Include clear labels indicating which plots are from ablation vs baseline",
    "5. Use consistent naming conventions for saved files",
]

DETERMINE_DATASETS_INTRO = (
    "You are an AI researcher analyzing experiment results. Based on the plot analyses and feedback, determine which datasets are successfully tested. Return reasoning and the dataset names that are successfully executed, or an empty string if no datasets are successfully executed."
)

DETERMINE_DATASETS_RESPONSE_FORMAT = (
    "Your response should start with 'REASONING: <reasoning>' to think about the plot analysis and feedback in the first line."
    "In the second line, you should have a list of dataset names that are successfully executed, starting with 'SUCCESSFULLY_TESTED_DATASETS: <list_datasets_successfully_tested>', "
)

PLOT_SELECTION_INTRO_TEMPLATE = (
    "You are an experienced AI researcher analyzing experimental results. "
    "You have been provided with plots from a machine learning experiment. "
    "Please select {max_plots} most relevant plots to analyze. "
    "For similar plots (e.g. generated samples at each epoch), select only at most {max_similar_plots} plots at a suitable interval of epochs."
    "Format your response as a list of plot paths, where each plot path includes the full path to the plot file."
)

VLM_USER_MESSAGE_TEMPLATE = (
    "You are an experienced AI researcher analyzing experimental results. "
    "You have been provided with plots from a machine learning experiment. "
    "This experiment is based on the following research idea: {task_desc}"
    "Please analyze these plots and provide detailed insights about the results. "
    "If you don't receive any plots, say 'No plots received'. "
    "Never make up plot analysis. "
    "Please return the analyzes with strict order of uploaded images, but DO NOT include any word "
    "like 'the first plot'."
)

NODE_SUMMARY_INTRO = (
    "You are an AI researcher analyzing experimental results. "
    "Please summarize the findings from this experiment iteration."
)

GLOBAL_METRICS_INTRO = (
    "You are an AI researcher setting up experiments. "
    "Please propose meaningful evaluation metrics that will help analyze "
    "the performance and characteristics of solutions for this research task."
)

GLOBAL_METRICS_INSTRUCTIONS = [
    "Propose a single evaluation metric that would be useful for analyzing the performance of solutions for this research task.",
    "Note: Validation loss will be tracked separately so you don't need to include it in your response.",
    "Format your response as a list containing:",
    "- name: The name of the metric",
    "- maximize: Whether higher values are better (true/false)",
    "- description: A brief explanation of what the metric measures"
    "Your list should contain only one metric.",
]

PARSE_METRICS_INTRO = (
    "You are an AI researcher analyzing experimental results stored in numpy files. "
    "Write code to load and analyze the metrics from experiment_data.npy."
)

PARSE_METRICS_INSTRUCTIONS = [
    "0. Make sure to get the working directory from os.path.join(os.getcwd(), 'working')",
    "1. Load the experiment_data.npy file, which is located in the working directory",
    "2. Extract metrics for each dataset. Make sure to refer to the original code to understand the structure of the data.",
    "3. Always print the name of the dataset before printing the metrics",
    "4. Always print the name of the metric before printing the value by specifying the metric name clearly. Avoid vague terms like 'train,' 'val,' or 'test.' Instead, use precise labels such as 'train accuracy,' 'validation loss,' or 'test F1 score,' etc.",
    "5. You only need to print the best or final value for each metric for each dataset",
    "6. DO NOT CREATE ANY PLOTS",
    "Important code structure requirements:",
    "  - Do NOT put any execution code inside 'if __name__ == \"__main__\":' block. Do not use 'if __name__ == \"__main__\":' at all.",
    "  - All code should be at the global scope or in functions that are called from the global scope",
    "  - The script should execute immediately when run, without requiring any special entry point",
]

PARSE_METRICS_EXAMPLE_CODE = [
    """
                            import matplotlib.pyplot as plt
                            import numpy as np

                            experiment_data = np.load(os.path.join(os.getcwd(), 'experiment_data.npy'), allow_pickle=True).item()
                            """,
]

METRICS_PROMPT_INTRO = (
    "Parse the metrics from the execution output. You only need the final or best value of a metric for each dataset, not the entire list during training."
)

PARSE_EXEC_RESULT_INTRO = (
    "You are an experienced AI researcher. "
    "You have written code for your research experiment and now need to evaluate the output of the code execution. "
    "Analyze the execution output, determine if there were any bugs, and provide a summary of the findings. "
)

HYPERPARAM_TUNING_PROMPT_INTRO = (
    "You are an AI researcher conducting hyperparameter tuning for baseline experiments. "
    "Based on the current implementation and previous hyperparameter tuning attempts (if any), "
    "propose ONE new hyperparameter tuning idea to see if it improves the performance."
    "You should first check if simply training longer (more epochs) improves the performance."
    "Then try tuning common hyperparameters such as learning rate, batch size, etc."
    "Only propose algorithm-specific and/or model-specific hyperparameters after you have tried the above."
)

HYPERPARAM_TUNING_INSTRUCTIONS = [
    "1. Identify ONE specific hyperparameter to tune",
    "2. Ensure the hyperparameter is different from previous attempts",
]

HYPERPARAM_IDEA_RESPONSE_FORMAT = (
    "Your response should start with 'HYPERPARAM NAME: <hyperparam name>' on the first line to represent the name of the hyperparameter."
    "The second line should start with 'DESCRIPTION: <description>', a brief description of what hyperparameter is being tuned and why (3-5 sentences). "
)

ABLATION_PROMPT_INTRO = (
    "You are an AI researcher conducting ablation studies. "
    "Based on the current implementation and previous ablations (if any), "
    "propose ONE new ablation study that tests a different aspect of the model."
)

ABLATION_INSTRUCTIONS = [
    "1. Identify ONE specific component/feature to ablate",
    "2. Ensure the ablation is different from previous completed or running attempts",
    "3. The ablation should be a new idea, not a variation of previous ideas",
    "4. If you have only used a single synthetic dataset throughout the experiment, one of your ablations should be to use multiple synthetic datasets (at least 3 different datasets)",
]

ABLATION_PROMPT_RESPONSE_FORMAT = (
    "Your response should start with 'ABLATION NAME: <ablation name>' on the first line to represent the name of the ablation."
    "The second line should start with 'ABLATION DESCRIPTION: <description>', a brief description of what component is being ablated and why (3-5 sentences), "
)

SEED_AGG_PLOTTING_GUIDELINE = [
    "REQUIREMENTS: ",
    "The code should start with:",
    "  import matplotlib.pyplot as plt",
    "  import numpy as np",
    "  import os",
    "  working_dir = os.path.join(os.getcwd(), 'working')",
    "Create standard visualizations of experiment results",
    "Save all plots to working_dir",
    "Include training/validation curves if available",
    "ONLY plot data that exists in experiment_data.npy - DO NOT make up or simulate any values",
    "Use basic matplotlib without custom styles",
    "Each plot should be in a separate try-except block",
    "Always close figures after saving",
    "Always include a title for each plot, and be sure to use clear subtitles—such as 'Left: Ground Truth, Right: Generated Samples'—while also specifying the type of dataset being used.",
    "Make sure to use descriptive names for figures when saving e.g. always include the dataset name and the type of plot in the name",
    "When there are many similar figures to plot (e.g. generated samples at each epoch), make sure to plot only at a suitable interval of epochs so that you only plot at most 5 figures.",
    "Example to extract data from experiment_data: experiment_data['dataset_name_1']['metrics']['train']",
    "Make sure to add legend for standard error bars and means if applicable",
    "Example data loading and plot saving code: ",
    """
                try:
                    experiment_data_path_list = # Make sure to use the correct experiment data path that's provided in the Experiment Data Path section
                    all_experiment_data = []
                    for experiment_data_path in experiment_data_path_list:
                        experiment_data = np.load(os.path.join(os.getenv(\"AI_SCIENTIST_ROOT\"), experiment_data_path), allow_pickle=True).item()
                        all_experiment_data.append(experiment_data)
                except Exception as e:
                    print(f'Error loading experiment data: {e}')

                try:
                    # First plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_1].png')
                    plt.close()
                except Exception as e:
                    print(f\"Error creating plot1: {e}\")
                    plt.close()  # Always close figure even if error occurs

                try:
                    # Second plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_2].png')
                    plt.close()
                except Exception as e:
                    print(f\"Error creating plot2: {e}\")
                    plt.close()
            """,
]

SEED_AGG_PLOTTING_INTRO = (
    "You are an expert in data visualization and plotting. "
    "You are given a set of evaluation results and the code that was used to plot them. "
    "Your task is to write a new plotting code that aggregate the results "
    "e.g. for example, by adding mean values and standard error bars to the plots."
)

SEED_AGG_RESPONSE_FORMAT = (
    "Your response should be a brief outline/sketch of your proposed solution in natural language (7-10 sentences), "
    "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric(s) if applicable. "
    "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
)

PROMPTS = {
    "response_format": RESPONSE_FORMAT,
    "metricparse_response_format": METRICPARSE_RESPONSE_FORMAT,
    "debug_response_format": DEBUG_RESPONSE_FORMAT,
    "hyperparam_code_response_format": HYPERPARAM_CODE_RESPONSE_FORMAT,
    "ablation_code_response_format": ABLATION_RESPONSE_FORMAT,
    "environment_prompt_template": ENVIRONMENT_PROMPT_TEMPLATE,
    "impl_guideline_base": IMPL_GUIDELINE_BASE,
    "impl_guideline_suffix": IMPL_GUIDELINE_SUFFIX,
    "impl_guideline_kfold_template": IMPL_GUIDELINE_KFOLD_TEMPLATE,
    "impl_guideline_multi_dataset": IMPL_GUIDELINE_MULTI_DATASET,
    "draft_intro": DRAFT_INTRO,
    "draft_experiment_guideline": DRAFT_EXPERIMENT_GUIDELINE,
    "debug_intro": DEBUG_INTRO,
    "debug_guideline": DEBUG_GUIDELINE,
    "improve_intro": IMPROVE_INTRO,
    "hyperparam_node_intro_template": HYPERPARAM_INTRO_TEMPLATE,
    "hyperparam_node_guideline": HYPERPARAM_GUIDELINE,
    "ablation_node_intro_template": ABLATION_INTRO_TEMPLATE,
    "ablation_node_guideline": ABLATION_GUIDELINE,
    "plotting_guideline_base": PLOTTING_GUIDELINE_BASE,
    "plotting_guideline_stage3": PLOTTING_GUIDELINE_STAGE3,
    "plotting_guideline_stage4": PLOTTING_GUIDELINE_STAGE4,
    "determine_datasets_intro": DETERMINE_DATASETS_INTRO,
    "determine_datasets_response_format": DETERMINE_DATASETS_RESPONSE_FORMAT,
    "plot_selection_intro_template": PLOT_SELECTION_INTRO_TEMPLATE,
    "vlm_user_message_template": VLM_USER_MESSAGE_TEMPLATE,
    "node_summary_intro": NODE_SUMMARY_INTRO,
    "global_metrics_intro": GLOBAL_METRICS_INTRO,
    "global_metrics_instructions": GLOBAL_METRICS_INSTRUCTIONS,
    "parse_metrics_intro": PARSE_METRICS_INTRO,
    "parse_metrics_instructions": PARSE_METRICS_INSTRUCTIONS,
    "parse_metrics_example_code": PARSE_METRICS_EXAMPLE_CODE,
    "metrics_prompt_intro": METRICS_PROMPT_INTRO,
    "parse_exec_result_intro": PARSE_EXEC_RESULT_INTRO,
    "hyperparam_idea_intro": HYPERPARAM_TUNING_PROMPT_INTRO,
    "hyperparam_idea_instructions": HYPERPARAM_TUNING_INSTRUCTIONS,
    "hyperparam_idea_response_format": HYPERPARAM_IDEA_RESPONSE_FORMAT,
    "ablation_idea_intro": ABLATION_PROMPT_INTRO,
    "ablation_idea_instructions": ABLATION_INSTRUCTIONS,
    "ablation_idea_response_format": ABLATION_PROMPT_RESPONSE_FORMAT,
    "seed_agg_plotting_guideline": SEED_AGG_PLOTTING_GUIDELINE,
    "seed_agg_plotting_intro": SEED_AGG_PLOTTING_INTRO,
    "seed_agg_response_format": SEED_AGG_RESPONSE_FORMAT,
}
