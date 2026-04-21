import os
import yaml
import pandas as pd


def save_dataframe_to_csv_with_metadata(dataframe: pd.DataFrame, save_path: str) -> None:
    # Split save path into root and file name
    save_path_root, save_file_name = os.path.split(save_path)
    
    # Remove constant columns and convert them to metadata
    constant_columns = []
    metadata = dict()
    for column in dataframe.columns:
        if len(dataframe[column].unique()) == 1:
            constant_columns.append(column)
            value = list(dataframe[column].unique())[0]
            try:
                value = value.item()
            except AttributeError:
                pass
            metadata[column] = value

    dataframe = dataframe.drop(columns=constant_columns)

    # Save metadata as yaml
    metadata_save_file_name = f"metadata_{save_file_name}".replace(".csv", ".yaml")
    metadata_save_path = os.path.join(save_path_root, metadata_save_file_name)
    with open(metadata_save_path, "w") as file:
        yaml.dump(metadata, file)
    
    # Drop prompts
    if "prompt" in dataframe.columns:
        dataframe = dataframe.drop(columns=["prompt"])
    # Drop descriptions
    if "description" in dataframe.columns:
        dataframe = dataframe.drop(columns=["description"])
    # Simplify "name" column
    if "name" in dataframe.columns:
        dataframe["name"] = dataframe["name"].apply(lambda name: name.split("/")[-1])
    
    # Save results
    dataframe.to_csv(save_path, index=False)
    return None


def load_dataframe_from_csv_with_metadata(load_path: str) -> pd.DataFrame:
    # Split load path into root and file name
    load_path_root, load_file_name = os.path.split(load_path)
    
    # Load metadata
    metadata_load_file_name = f"metadata_{load_file_name}".replace(".csv", ".yaml")
    metadata_load_path = os.path.join(load_path_root, metadata_load_file_name)
    with open(metadata_load_path, "r") as file:
        metadata = yaml.load(file, Loader=yaml.FullLoader)
    
    # Load dataframe
    dataframe = pd.read_csv(load_path)
    
    # Add metadata to dataframe
    for key, value in metadata.items():
        dataframe[key] = value
    
    return dataframe
