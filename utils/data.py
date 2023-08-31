import pandas as pd


def simplify_data(csv_data_path="rotten_tomatoes_movies.csv"):
    df = pd.read_csv(csv_data_path)
    print(df.head().T)

    selected_columns = [
        "rotten_tomatoes_link",
        "movie_title",
        "movie_info",
        "critics_consensus",
        "genres",
        "original_release_date",
        "tomatometer_status",
        "audience_status",
    ]
    selected_columns_df = df[selected_columns]

    # Sort the DataFrame by "original_release_date"
    sorted_df = selected_columns_df.sort_values(
        by="original_release_date", ascending=False
    )

    # Filter rows with an original_release_date of 1970 and newer
    filtered_df = sorted_df[sorted_df["original_release_date"] >= "2018-01-01"]

    final_df = filtered_df[
        (filtered_df["tomatometer_status"].isin(["Fresh", "Certified-Fresh"]))
        & (filtered_df["audience_status"] == "Upright")
    ]
    return final_df
