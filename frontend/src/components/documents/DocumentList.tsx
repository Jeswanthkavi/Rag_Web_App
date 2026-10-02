import React, { useState } from "react";
import { api } from "../../lib/api";
import { APIError, DocumentItem, DocumentStatus } from "../../types";

interface DocumentListProps {
  documents: DocumentItem[];
  isLoading: boolean;
  onRefresh: () => void;
  onDeleteSuccess: (docId: string) => void;
}

export const DocumentList: React.FC<DocumentListProps> = ({
  documents,
  isLoading,
  onRefresh,
  onDeleteSuccess,
}) => {
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const handleDelete = async (doc: DocumentItem) => {
    if (!window.confirm(`Are you sure you want to delete '${doc.filename}'? This will purge its chunks and vector embeddings.`)) {
      return;
    }

    setDeletingId(doc.id);
    setActionError(null);
    try {
      await api.deleteDocument(doc.id);
      onDeleteSuccess(doc.id);
    } catch (err: unknown) {
      const apiErr = err as APIError;
      setActionError(apiErr.detail || "Failed to delete document.");
    } finally {
      setDeletingId(null);
    }
  };

  const getStatusBadge = (status: DocumentStatus) => {
    switch (status) {
      case "COMPLETED":
        return <span className="badge badge-success">✓ Completed</span>;
      case "PROCESSING":
        return <span className="badge badge-warning animate-pulse">⏳ Processing</span>;
      case "PENDING":
        return <span className="badge badge-info">⏱ Pending</span>;
      case "FAILED":
        return <span className="badge badge-error">✕ Failed</span>;
      default:
        return <span className="badge badge-info">{status}</span>;
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (dateStr: string) => {
    try {
      return new Date(dateStr).toLocaleString();
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="glass-panel" style={{ padding: "1.75rem" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "1.25rem",
        }}
      >
        <div>
          <h2 style={{ fontSize: "1.1rem", fontWeight: 600 }}>Your Documents</h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
            {documents.length} document{documents.length === 1 ? "" : "s"} indexed in your tenant corpus
          </p>
        </div>
        <button
          onClick={onRefresh}
          className="btn-secondary"
          disabled={isLoading}
          style={{ padding: "0.45rem 0.9rem", fontSize: "0.85rem" }}
        >
          🔄 Refresh
        </button>
      </div>

      {actionError && (
        <div
          style={{
            background: "var(--status-error-bg)",
            color: "var(--status-error)",
            border: "1px solid rgba(239, 68, 68, 0.3)",
            padding: "0.75rem 1rem",
            borderRadius: "var(--radius-md)",
            fontSize: "0.85rem",
            marginBottom: "1rem",
          }}
        >
          ⚠️ {actionError}
        </div>
      )}

      {isLoading && documents.length === 0 ? (
        <div
          style={{
            textAlign: "center",
            padding: "3rem 1rem",
            color: "var(--text-muted)",
          }}
        >
          <div
            style={{
              width: "28px",
              height: "28px",
              border: "3px solid rgba(99, 102, 241, 0.2)",
              borderTopColor: "var(--accent-primary)",
              borderRadius: "50%",
              margin: "0 auto 1rem",
            }}
            className="animate-spin"
          />
          <p>Loading document catalog...</p>
        </div>
      ) : documents.length === 0 ? (
        <div
          style={{
            textAlign: "center",
            padding: "3rem 1rem",
            color: "var(--text-muted)",
            background: "rgba(0, 0, 0, 0.2)",
            borderRadius: "var(--radius-md)",
          }}
        >
          <p style={{ fontSize: "1.5rem", marginBottom: "0.5rem" }}>📂</p>
          <p style={{ fontWeight: 500, color: "var(--text-secondary)", marginBottom: "0.25rem" }}>
            No documents uploaded yet
          </p>
          <p style={{ fontSize: "0.85rem" }}>
            Upload PDFs, DOCX, or TXT files above to begin grounding your RAG queries.
          </p>
        </div>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: "0.9rem",
              textAlign: "left",
            }}
          >
            <thead>
              <tr
                style={{
                  borderBottom: "1px solid var(--border-subtle)",
                  color: "var(--text-muted)",
                  fontSize: "0.8rem",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                <th style={{ padding: "0.75rem 1rem" }}>Filename</th>
                <th style={{ padding: "0.75rem 1rem" }}>Type</th>
                <th style={{ padding: "0.75rem 1rem" }}>Size</th>
                <th style={{ padding: "0.75rem 1rem" }}>Chunks</th>
                <th style={{ padding: "0.75rem 1rem" }}>Status</th>
                <th style={{ padding: "0.75rem 1rem" }}>Uploaded</th>
                <th style={{ padding: "0.75rem 1rem", textAlign: "right" }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr
                  key={doc.id}
                  style={{
                    borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
                    transition: "background 0.2s ease",
                  }}
                >
                  <td style={{ padding: "0.85rem 1rem", fontWeight: 500 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <span>{doc.file_type === ".pdf" ? "📕" : doc.file_type === ".docx" ? "📘" : "📄"}</span>
                      <span>{doc.filename}</span>
                    </div>
                  </td>
                  <td style={{ padding: "0.85rem 1rem", color: "var(--text-muted)" }}>
                    <span className="code-inline">{doc.file_type}</span>
                  </td>
                  <td style={{ padding: "0.85rem 1rem", color: "var(--text-secondary)" }}>
                    {formatBytes(doc.file_size)}
                  </td>
                  <td style={{ padding: "0.85rem 1rem", color: "var(--text-accent)", fontWeight: 600 }}>
                    {doc.total_chunks || 0}
                  </td>
                  <td style={{ padding: "0.85rem 1rem" }}>
                    {getStatusBadge(doc.status)}
                    {doc.error_message && (
                      <div style={{ fontSize: "0.75rem", color: "var(--status-error)", marginTop: "0.25rem" }}>
                        {doc.error_message}
                      </div>
                    )}
                  </td>
                  <td style={{ padding: "0.85rem 1rem", color: "var(--text-muted)", fontSize: "0.8rem" }}>
                    {formatDate(doc.created_at)}
                  </td>
                  <td style={{ padding: "0.85rem 1rem", textAlign: "right" }}>
                    <button
                      onClick={() => handleDelete(doc)}
                      disabled={deletingId === doc.id}
                      className="btn-danger"
                      style={{ padding: "0.35rem 0.75rem", fontSize: "0.8rem" }}
                    >
                      {deletingId === doc.id ? "Deleting..." : "🗑 Delete"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
