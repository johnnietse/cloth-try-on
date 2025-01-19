-- Table to store video metadata
-- CREATE TABLE videos (
--     id SERIAL PRIMARY KEY,
--     original_path TEXT NOT NULL,
--     processed_path TEXT,
--     upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
--     status TEXT DEFAULT 'uploaded', -- 'uploaded', 'processing', 'processed', 'failed'
--     error_message TEXT
-- );

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

CREATE TABLE videos (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255),
    processed_filename VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);