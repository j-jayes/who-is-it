import os
import pandas as pd


def load_and_merge_data(input_file_1, input_file_2, columns_to_select):
    """
    Load two parquet files, select specific columns from the second file, and merge them by columns.
    """
    df_1 = pd.read_parquet(input_file_1)
    df_2 = pd.read_parquet(input_file_2)
    df_2 = df_2[columns_to_select]
    return pd.concat([df_1, df_2], axis=1)


def save_to_parquet(df, output_dir, output_filename):
    """
    Save the dataframe to a parquet file in the specified directory.
    Creates the directory if it doesn't exist.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_filename)
    df.to_parquet(output_file, index=False)
    print(f"File saved to {output_file}")


def process_occupations(language):
    """
    Process and merge occupation and classification data for a specified language.
    """
    base_dir = "data/occupations_for_classification"
    input_file_1 = os.path.join(base_dir, f"{language}_occupations.parquet")
    input_file_2 = os.path.join(base_dir, f"{language}_occupations_classified.parquet")
    output_dir = os.path.join(base_dir, "occupations_classified")
    output_filename = f"{language}_occupations_classified_joined.parquet"

    columns_to_select = ["inputs", "hisco_1", "prob_1", "desc_1"]
    merged_data = load_and_merge_data(input_file_1, input_file_2, columns_to_select)
    save_to_parquet(merged_data, output_dir, output_filename)


def main():
    """
    Main function to process occupations for multiple languages.
    """
    languages = ["english", "swedish"]
    for language in languages:
        process_occupations(language)


if __name__ == "__main__":
    main()
