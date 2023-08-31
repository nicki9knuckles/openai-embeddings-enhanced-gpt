import requests
import json
import pickle
import pandas as pd


def fetch_movies_from_tmdb(api_key, year):
    url = "https://api.themoviedb.org/3/discover/movie"
    params = {"api_key": api_key, "primary_release_year": year}

    all_movies = []
    current_page = 1
    total_pages = 1  # Placeholder, will be updated from the first response

    while current_page <= total_pages:
        params["page"] = current_page
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = json.loads(response.text)
            all_movies.extend(data["results"])
            total_pages = data["total_pages"]
            current_page += 1
        else:
            print(f"Failed to get data: {response.status_code}")
            break

    return all_movies


def main():
    api_key = "a52ddbdad4366bc1c15432a0e6f5627a"
    year = 2023
    pickle_file_path = "tmdb_movies_2023.pkl"

    # Try to load from pickle file
    try:
        with open(pickle_file_path, "rb") as f:
            movies = pickle.load(f)
    except FileNotFoundError:
        movies = fetch_movies_from_tmdb(api_key, year)
        # Save to pickle file
        with open(pickle_file_path, "wb") as f:
            pickle.dump(movies, f)

    # print(movies)

    with open("tmdb_movies_2023.pkl", "rb") as file:
        data_list = pickle.load(file)

    df = pd.DataFrame(data_list)

    print(df.head().T)


if __name__ == "__main__":
    main()
