from __future__ import annotations

from KNNModel import build_model, recommend_similar_songs, recommend_for_user


def run_cli():
    df, scaler, knn, X_scaled = build_model()

    while True:
        print("\n=== Spotify KNN Recommender ===")
        print("1) Find songs similar to one track")
        print("2) Get playlist from your liked songs")
        print("3) Quit")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            title = input("Song title: ").strip()
            artist = input("Artist (optional): ").strip() or None
            try:
                recs = recommend_similar_songs(
                    df, scaler, knn, X_scaled, title, artist, top_k=10
                )
                print("\nRecommendations:\n")
                print(recs.to_string(index=False))
            except ValueError as e:
                print("Error:", e)

        elif choice == "2":
            print("Enter liked songs (blank title to finish).")
            liked: list[tuple[str, str | None]] = []
            while True:
                t = input("Song title (blank to stop): ").strip()
                if not t:
                    break
                a = input("Artist (optional): ").strip() or None
                liked.append((t, a))

            if not liked:
                print("No songs entered.")
                continue

            # Simple example filters; you can tweak/remove as you like
            try:
                top_k = int(input("How many recommendations? (default 15): ") or "15")
            except ValueError:
                top_k = 15

            use_filters = input(
                "Apply simple filters (year >= 2000, popularity >= 40, non-explicit only)? [y/N]: "
            ).strip().lower()

            min_year = 2000 if use_filters == "y" else None
            min_popularity = 40 if use_filters == "y" else None
            explicit_only = False if use_filters == "y" else None

            try:
                recs = recommend_for_user(
                    df,
                    scaler,
                    knn,
                    X_scaled,
                    liked,
                    top_k=top_k,
                    max_per_artist=2,
                    min_year=min_year,
                    min_popularity=min_popularity,
                    explicit_only=explicit_only,
                )
                print("\nYour playlist recommendations:\n")
                print(recs.to_string(index=False))
            except ValueError as e:
                print("Error:", e)

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please pick 1, 2, or 3.")


if __name__ == "__main__":
    run_cli()
