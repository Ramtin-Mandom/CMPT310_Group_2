from __future__ import annotations

import sys

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QTabWidget,
    QSpinBox,
    QTableWidgetSelectionRange,
)
from PyQt6.QtCore import Qt

import KNNModel  # make sure KNNModel.py is in the same folder


class RecommenderWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spotify KNN Recommender")

        # store liked tracks as list[(title, artist|None)]
        self.liked_tracks: list[tuple[str, str | None]] = []

        # ---- Load your model once at startup ----
        try:
            self.df = KNNModel.load_and_prepare(KNNModel.CSV_PATH)
            self.scaler, self.knn, self.X_scaled = KNNModel.fit_scaler_and_knn(
                self.df,
                n_neighbors=KNNModel.N_NEIGHBORS,
                metric=KNNModel.METRIC,
            )
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", f"Could not load model:\n{e}")
            raise

        self._build_ui()
        self._apply_styles()

    # ------------------------------------------------------------------
    # UI Layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel("Spotify KNN Recommender")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        main_layout.addWidget(title_label)

        subtitle = QLabel("Discover songs and build playlists from your favourites")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        subtitle.setObjectName("SubtitleLabel")
        main_layout.addWidget(subtitle)

        # Tabs: (1) Similar to one song, (2) Playlist from liked songs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_song_tab(), "Similar to one song")
        self.tabs.addTab(self._build_user_tab(), "From liked songs")
        main_layout.addWidget(self.tabs)

        results_label = QLabel("Recommendations")
        results_label.setObjectName("SectionHeader")
        main_layout.addWidget(results_label)

        # Table to show recommendations
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Rank", "Name", "Artists", "Similarity"])
        self.table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.table)

        self.setLayout(main_layout)
        self.resize(950, 650)

    def _build_song_tab(self) -> QWidget:
        """Tab 1: recommend similar songs to a single track."""
        w = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)

        info = QLabel("Type a song and optionally its artist to find similar tracks.")
        info.setObjectName("TabInfoLabel")
        layout.addWidget(info)

        # Song title row
        row_title = QHBoxLayout()
        lbl_title = QLabel("Song title:")
        lbl_title.setObjectName("FieldLabel")
        row_title.addWidget(lbl_title)

        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("e.g., Back In Black")
        row_title.addWidget(self.title_edit)

        layout.addLayout(row_title)

        # Artist (optional) row
        row_artist = QHBoxLayout()
        lbl_artist = QLabel("Artist (optional):")
        lbl_artist.setObjectName("FieldLabel")
        row_artist.addWidget(lbl_artist)

        self.artist_edit = QLineEdit()
        self.artist_edit.setPlaceholderText("e.g., AC/DC")
        row_artist.addWidget(self.artist_edit)

        layout.addLayout(row_artist)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn = QPushButton("Recommend similar songs")
        btn.clicked.connect(self.on_song_recommend_clicked)
        btn_row.addWidget(btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        w.setLayout(layout)
        return w

    def _build_user_tab(self) -> QWidget:
        """Tab 2: recommend playlist based on liked songs."""
        w = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)

        instructions = QLabel(
            "Build your liked song list, then let the recommender create a playlist.\n"
            "Add songs one by one using the fields below."
        )
        instructions.setObjectName("TabInfoLabel")
        layout.addWidget(instructions)

        # Input row for a single liked song
        input_row = QHBoxLayout()

        lbl_liked_title = QLabel("Title:")
        lbl_liked_title.setObjectName("FieldLabel")
        input_row.addWidget(lbl_liked_title)

        self.liked_title_edit = QLineEdit()
        self.liked_title_edit.setPlaceholderText("e.g., Back In Black")
        input_row.addWidget(self.liked_title_edit)

        lbl_liked_artist = QLabel("Artist (optional):")
        lbl_liked_artist.setObjectName("FieldLabel")
        input_row.addWidget(lbl_liked_artist)

        self.liked_artist_edit = QLineEdit()
        self.liked_artist_edit.setPlaceholderText("e.g., AC/DC")
        input_row.addWidget(self.liked_artist_edit)

        self.add_liked_btn = QPushButton("Add to liked list")
        self.add_liked_btn.clicked.connect(self.on_add_liked_clicked)
        input_row.addWidget(self.add_liked_btn)

        layout.addLayout(input_row)

        # Table to show liked tracks
        liked_label = QLabel("Liked songs")
        liked_label.setObjectName("SectionHeaderSmall")
        layout.addWidget(liked_label)

        self.liked_table = QTableWidget(0, 2)
        self.liked_table.setHorizontalHeaderLabels(["Title", "Artist"])
        self.liked_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.liked_table)

        # Controls row: clear list, number of recs, recommend button
        controls_row = QHBoxLayout()

        self.clear_liked_btn = QPushButton("Clear liked list")
        self.clear_liked_btn.clicked.connect(self.on_clear_liked_clicked)
        controls_row.addWidget(self.clear_liked_btn)

        controls_row.addStretch()

        lbl_num = QLabel("Number of recommendations:")
        lbl_num.setObjectName("FieldLabel")
        controls_row.addWidget(lbl_num)

        self.num_spin = QSpinBox()
        self.num_spin.setRange(1, 50)
        self.num_spin.setValue(15)
        controls_row.addWidget(self.num_spin)

        recommend_btn = QPushButton("Recommend playlist")
        recommend_btn.clicked.connect(self.on_user_recommend_clicked)
        controls_row.addWidget(recommend_btn)

        layout.addLayout(controls_row)

        w.setLayout(layout)
        return w

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _apply_styles(self):
        """
        Apply a light / green / white theme and nicer fonts.
        """
        self.setStyleSheet(
            """
            QWidget {
                background-color: #f5fff7;
                font-family: 'Segoe UI', 'Arial';
                font-size: 11pt;
                color: #064420;
            }

            #TitleLabel {
                font-size: 22pt;
                font-weight: 700;
                color: #064420;
            }

            #SubtitleLabel {
                font-size: 10pt;
                color: #2c6e49;
            }

            #SectionHeader {
                font-size: 13pt;
                font-weight: 600;
                margin-top: 10px;
                margin-bottom: 4px;
                color: #064420;
            }

            #SectionHeaderSmall {
                font-size: 11pt;
                font-weight: 600;
                margin-top: 6px;
                margin-bottom: 2px;
                color: #064420;
            }

            #FieldLabel {
                font-weight: 600;
                color: #0b5d3e;
            }

            #TabInfoLabel {
                color: #2c6e49;
            }

            QTabWidget::pane {
                border: 1px solid #9adfbc;
                border-radius: 8px;
                margin-top: 6px;
            }

            QTabBar::tab {
                background: #e0f7ea;
                padding: 8px 14px;
                margin: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                color: #064420;
                font-weight: 500;
            }

            QTabBar::tab:selected {
                background: #2ecc71;
                color: white;
            }

            QTabBar::tab:hover {
                background: #baf3cc;
            }

            QPushButton {
                background-color: #2ecc71;
                color: white;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 600;
                border: none;
            }

            QPushButton:hover {
                background-color: #27ae60;
            }

            QPushButton:pressed {
                background-color: #1e8449;
            }

            QLineEdit, QSpinBox {
                border: 1px solid #9adfbc;
                border-radius: 5px;
                padding: 4px 6px;
                background: white;
            }

            QTableWidget {
                background: white;
                border: 1px solid #c8e6d4;
                gridline-color: #d5f5e3;
            }

            QHeaderView::section {
                background-color: #d5f5e3;
                padding: 4px;
                border: none;
                font-weight: 600;
            }
            """
        )

    # ------------------------------------------------------------------
    # Helper to show DataFrame results in the table
    # ------------------------------------------------------------------
    def show_recommendations(self, df):
        self.table.setRowCount(len(df))
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Rank", "Name", "Artists", "Similarity"])

        for row_idx, (_, row) in enumerate(df.iterrows()):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(row.get("rank", ""))))
            self.table.setItem(row_idx, 1, QTableWidgetItem(str(row.get("name", ""))))
            self.table.setItem(row_idx, 2, QTableWidgetItem(str(row.get("artists", ""))))
            self.table.setItem(row_idx, 3, QTableWidgetItem(str(row.get("similarity", ""))))

        self.table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Button handlers – Tab 1
    # ------------------------------------------------------------------
    def on_song_recommend_clicked(self):
        title = self.title_edit.text().strip()
        artist = self.artist_edit.text().strip() or None

        if not title:
            QMessageBox.warning(self, "Missing title", "Please enter a song title.")
            return

        try:
            recs = KNNModel.recommend_similar_songs(
                self.df,
                self.scaler,
                self.knn,
                self.X_scaled,
                query_title=title,
                query_artist=artist,
                top_k=10,
            )
        except ValueError as e:
            QMessageBox.warning(self, "Song not found", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error:\n{e}")
            return

        self.show_recommendations(recs)

    # ------------------------------------------------------------------
    # Button handlers – Tab 2
    # ------------------------------------------------------------------
    def on_add_liked_clicked(self):
        title = self.liked_title_edit.text().strip()
        artist = self.liked_artist_edit.text().strip() or None

        if not title:
            QMessageBox.warning(self, "Missing title", "Please enter a song title.")
            return

        # add to internal list
        self.liked_tracks.append((title, artist))

        # update table
        row = self.liked_table.rowCount()
        self.liked_table.insertRow(row)
        self.liked_table.setItem(row, 0, QTableWidgetItem(title))
        self.liked_table.setItem(row, 1, QTableWidgetItem(artist or ""))

        # clear inputs for next entry
        self.liked_title_edit.clear()
        self.liked_artist_edit.clear()
        self.liked_title_edit.setFocus()

    def on_clear_liked_clicked(self):
        self.liked_tracks.clear()
        self.liked_table.setRowCount(0)

    def on_user_recommend_clicked(self):
        if not self.liked_tracks:
            QMessageBox.warning(
                self,
                "No liked songs",
                "Please add at least one liked song before requesting a playlist.",
            )
            return

        top_k = self.num_spin.value()

        try:
            recs = KNNModel.recommend_for_user(
                self.df,
                self.scaler,
                self.knn,
                self.X_scaled,
                liked_tracks=self.liked_tracks,
                top_k=top_k,
            )
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error:\n{e}")
            return

        self.show_recommendations(recs)


def main():
    app = QApplication(sys.argv)
    win = RecommenderWindow()
    win.show()
    sys.exit(app.exec())
 

if __name__ == "__main__":
    main()
