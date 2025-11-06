-- Extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Extension for vector operations (for embeddings)
CREATE EXTENSION IF NOT EXISTS vector;

-- Texts table: stores text metadata and content
CREATE TABLE IF NOT EXISTS texts (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT,  -- For short texts (<10KB)
    file_path VARCHAR(1000),  -- For large texts (>10KB) stored in files
    length INTEGER NOT NULL DEFAULT 0,
    lines INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_texts_created_at ON texts(created_at DESC);
CREATE INDEX idx_texts_title ON texts(title);

-- Embeddings table: stores vector embeddings
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text_id VARCHAR(255) NOT NULL REFERENCES texts(id) ON DELETE CASCADE,
    model VARCHAR(100) NOT NULL,
    embedding vector(768),  -- Adjust dimension based on your model
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_embeddings_text_id ON embeddings(text_id);
CREATE INDEX idx_embeddings_model ON embeddings(model);
CREATE UNIQUE INDEX idx_embeddings_text_model ON embeddings(text_id, model);

-- Analysis History table: stores completed analyses
CREATE TABLE IF NOT EXISTS analysis_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type VARCHAR(50) NOT NULL,  -- semantic, style, tfidf, emotion, llm, combined, matrix
    text_id1 VARCHAR(255) REFERENCES texts(id) ON DELETE SET NULL,
    text_id2 VARCHAR(255) REFERENCES texts(id) ON DELETE SET NULL,
    text1_title VARCHAR(500),
    text2_title VARCHAR(500),
    similarity FLOAT,
    interpretation TEXT,
    details JSONB,  -- Store additional analysis details as JSON
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analysis_history_created_at ON analysis_history(created_at DESC);
CREATE INDEX idx_analysis_history_type ON analysis_history(type);
CREATE INDEX idx_analysis_history_text_ids ON analysis_history(text_id1, text_id2);

-- Tasks table: stores background task information
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed
    progress INTEGER DEFAULT 0,
    progress_message TEXT,
    metadata JSONB,
    result JSONB,
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_type ON tasks(type);
CREATE INDEX idx_tasks_created_at ON tasks(created_at DESC);

-- Cache table: stores frequently accessed data
CREATE TABLE IF NOT EXISTS cache (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cache_expires_at ON cache(expires_at);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updating updated_at
CREATE TRIGGER update_texts_updated_at BEFORE UPDATE ON texts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at BEFORE UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to clean up expired cache entries
CREATE OR REPLACE FUNCTION clean_expired_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE 'plpgsql';

-- Comments for documentation
COMMENT ON TABLE texts IS 'Stores text metadata and content (short texts in DB, large texts in files)';
COMMENT ON TABLE embeddings IS 'Stores vector embeddings for texts';
COMMENT ON TABLE analysis_history IS 'Stores completed text analysis results';
COMMENT ON TABLE tasks IS 'Stores background task information and results';
COMMENT ON TABLE cache IS 'Stores cached data with optional expiration';

-- Initial data (optional)
-- INSERT INTO texts (id, title, content, length, lines) VALUES
--     ('demo-text-1', 'Demo Text 1', 'This is a demo text for testing.', 30, 1),
--     ('demo-text-2', 'Demo Text 2', 'Another demo text for comparison.', 34, 1);
