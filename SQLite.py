import sqlite3


class SqliteTest:
    def __init__(self):
        self.connection = sqlite3.connect("veritabani.db")
        self.cursor = self.connection.cursor()
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS Patient (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            PatientID TEXT NOT NULL,
            PatientName TEXT,
            PatientStudy TEXT,
            PatientDate TEXT,
            PatientAge TEXT,
            PatientSex TEXT
        )
        """)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS Study (
            StudyID INTEGER PRIMARY KEY AUTOINCREMENT,
            StudyDescription TEXT,
            PatientID TEXT NOT NULL,
            FOREIGN KEY (PatientID) REFERENCES Patient(ID)
        )
        """)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS Series (
            SeriesID INTEGER PRIMARY KEY AUTOINCREMENT,
            SeriesDescription TEXT,
            StudyStudyID INTEGER NOT NULL,
            FOREIGN KEY (StudyStudyID) REFERENCES Study(StudyID)
        )
        """)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS DicomFile (
            DicomID INTEGER PRIMARY KEY AUTOINCREMENT,
            DicomName TEXT,
            DicomTags TEXT,
            SeriesSeriesID INTEGER NOT NULL,
            FOREIGN KEY (SeriesSeriesID) REFERENCES Series(SeriesID)
        )
        """)
        self.connection.commit()

    def num_patients(self):
        self.cursor.execute("SELECT COUNT(*) FROM Patient")
        return self.cursor.fetchone()[0]
    
    def num_studies(self):
        self.cursor.execute("SELECT COUNT(*) FROM Study")
        return self.cursor.fetchone()[0]
    
    def num_series(self):
        self.cursor.execute("SELECT COUNT(*) FROM Series")
        return self.cursor.fetchone()[0]
    
    def num_dicoms(self):
        self.cursor.execute("SELECT COUNT(*) FROM DicomFile")
        return self.cursor.fetchone()[0]

    def ekle_patient(self, id, name, study, date, age, sex):
        self.cursor.execute("""
        INSERT INTO Patient (PatientID, PatientName, PatientStudy, PatientDate, PatientAge, PatientSex)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (id, name, study, date, age, sex))
        self.connection.commit()

    def ekle_study(self, study, patId):
        self.cursor.execute("""
        INSERT INTO Study (StudyDescription, PatientID)
        VALUES (?, ?)
        """, (study, patId))
        self.connection.commit()

    def ekle_series(self, series, studyId):
        self.cursor.execute("""
        INSERT INTO Series (SeriesDescription, StudyStudyID)
        VALUES (?, ?)
        """, (series, studyId))
        self.connection.commit()
        
    def ekle_dicom(self, name, tags, seriesId):
        self.cursor.execute("""
        INSERT INTO DicomFile (DicomName, DicomTags, SeriesSeriesID)
        VALUES (?, ?, ?)
        """, (name, tags, seriesId))
        self.connection.commit()

    def allPatients(self, query):
        self.cursor.execute("SELECT * FROM Patient WHERE PatientName LIKE ? COLLATE NOCASE;", (query,))
        return self.cursor.fetchall()

    def listele_patient(self, id):
        self.cursor.execute("""
            SELECT D.DicomID, D.DicomName, D.DicomTags
            FROM DicomFile D
            JOIN Series S ON D.SeriesSeriesID = S.SeriesID
            JOIN Study ST ON S.StudyStudyID = ST.StudyID
            JOIN Patient P ON ST.PatientID = P.ID
            WHERE P.PatientID = ?;
        """, (id,))

        # Sonuçları al
        dicom_files = self.cursor.fetchall()

        return dicom_files
