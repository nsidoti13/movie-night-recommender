-- Movie Night Recommender — Database Schema

-- User accounts
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Stores each movie rating per user
CREATE TABLE IF NOT EXISTS ratings (
    user_name   TEXT    NOT NULL,
    movie_idx   INTEGER NOT NULL,
    status      TEXT    NOT NULL CHECK (status IN ('liked', 'disliked', 'unseen')),
    rated_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_name, movie_idx)
);

-- Stores per-user app state (done flag + current card)
CREATE TABLE IF NOT EXISTS user_state (
    user_name    TEXT    PRIMARY KEY,
    done         BOOLEAN DEFAULT FALSE,
    current_card INTEGER DEFAULT NULL,
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Handy view: summary per user
CREATE OR REPLACE VIEW rating_summary AS
SELECT
    user_name,
    COUNT(*) FILTER (WHERE status = 'liked')    AS liked,
    COUNT(*) FILTER (WHERE status = 'disliked') AS disliked,
    COUNT(*) FILTER (WHERE status = 'unseen')   AS unseen,
    COUNT(*)                                     AS total
FROM ratings
GROUP BY user_name;
