-- Create Database
CREATE DATABASE DSUHack;
USE DSUHack;

-- Users Table
CREATE TABLE Users (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Email VARCHAR(150) UNIQUE NOT NULL,
    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    Password VARCHAR(255) NOT NULL
);

-- Sensors Table
CREATE TABLE Sensors (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    UID INT NOT NULL,
    TestDateTime DATETIME NOT NULL,
    DurationSeconds INT NOT NULL,
    Delta FLOAT NOT NULL,
    Alpha FLOAT NOT NULL,
    Beta FLOAT NOT NULL,
    Gamma FLOAT NOT NULL,
    Spike INT NOT NULL,
    SpikeRate FLOAT NOT NULL,
    FOREIGN KEY (UID) REFERENCES Users(ID) ON DELETE CASCADE ON UPDATE CASCADE
);

-- Metrics Table
CREATE TABLE Metrics (
    MID INT AUTO_INCREMENT PRIMARY KEY,
    Strain FLOAT NOT NULL,
    Drift FLOAT NOT NULL,
    UID INT NOT NULL,
    RecoveryHint VARCHAR(255),
    FOREIGN KEY (UID) REFERENCES Users(ID) ON DELETE CASCADE ON UPDATE CASCADE
);


-- Insert Users
INSERT INTO Users (Name, Email, Password)
VALUES
('Alice Johnson', 'alice@example.com', 'hashed_pw_123'),
('Bob Smith', 'bob@example.com', 'hashed_pw_456'),
('Charlie Patel', 'charlie@example.com', 'hashed_pw_789');

-- Insert Sensor Data (Each linked to Users by UID)
INSERT INTO Sensors (UID, TestDateTime, DurationSeconds, Delta, Alpha, Beta, Gamma, Spike, SpikeRate)
VALUES
(1, '2025-09-10 10:15:00', 300, 1.2, 3.4, 2.1, 0.9, 5, 0.02),
(1, '2025-09-11 09:45:00', 280, 1.0, 3.1, 2.3, 1.1, 6, 0.03),
(2, '2025-09-10 14:20:00', 310, 1.5, 3.8, 2.5, 1.3, 8, 0.025),
(3, '2025-09-12 08:00:00', 295, 1.3, 3.2, 2.0, 1.0, 4, 0.015),
(3, '2025-09-13 07:50:00', 305, 1.1, 3.5, 2.2, 0.8, 7, 0.022);



-- Insert Metrics Data
INSERT INTO Metrics (Strain, Drift, UID, RecoveryHint)
VALUES
(0.42, 0.31, 1, 'Take a 5-min walk and hydrate.'),
(0.55, 0.40, 2, 'Try deep breathing for 2 minutes.'),
(0.30, 0.25, 3, 'Light stretch recommended.'),
(0.65, 0.50, 1, 'Schedule a recovery break soon.'),
(0.20, 0.15, 2, 'Great focus – keep going!');




