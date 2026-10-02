import React from "react";
import { Citation } from "../../types";

interface CitationModalProps {
  citation: Citation | null;
  onClose: () => void;
}

export const CitationModal: React.FC<CitationModalProps> = ({
  citation,
  onClose,
}) => {
  if (!citation) return null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0, 0, 0, 0.7)",
        backdropFilter: "blur(6px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 100,
        padding: "1rem",
      }}
      onClick={onClose}
    >
      <div
        className="glass-panel"
        style={{
          width: "100%",
          maxWidth: "560px",
          padding: "2rem",
          boxShadow: "var(--shadow-lg)",
          position: "relative",
          animation: "scaleUp 0.15s ease-out",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: "1.25rem",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span
              style={{
                background: "var(--accent-gradient)",
                color: "#ffffff",
                padding: "0.25rem 0.6rem",
                borderRadius: "var(--radius-sm)",
                fontSize: "0.85rem",
                fontWeight: 700,
              }}
            >
              [Source {citation.source_index}]
            </span>
            <span
              className={citation.verified ? "badge badge-success" : "badge badge-warning"}
            >
              {citation.verified ? "✓ Verified Grounding" : "Unverified"}
            </span>
          </div>
          <button
            onClick={onClose}
            style={{
              color: "var(--text-muted)",
              fontSize: "1.25rem",
              padding: "0.25rem",
            }}
          >
            ✕
          </button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", marginBottom: "1.5rem" }}>
          <div>
            <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
              Source Document:
            </span>
            <div style={{ fontWeight: 600, fontSize: "0.95rem", color: "var(--text-primary)" }}>
              {citation.filename}
            </div>
          </div>

          <div style={{ display: "flex", gap: "2rem" }}>
            {citation.page_number && (
              <div>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
                  Page:
                </span>
                <div style={{ fontWeight: 600, color: "var(--text-accent)" }}>
                  Page {citation.page_number}
                </div>
              </div>
            )}
            <div>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
                Chunk Position:
              </span>
              <div style={{ fontWeight: 600, color: "var(--text-secondary)" }}>
                Index {citation.chunk_index}
              </div>
            </div>
          </div>

          <div>
            <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
              Chunk ID:
            </span>
            <div className="code-inline" style={{ fontSize: "0.75rem", display: "inline-block" }}>
              {citation.chunk_id}
            </div>
          </div>
        </div>

        <div>
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", textTransform: "uppercase", display: "block", marginBottom: "0.4rem" }}>
            Retrieved Context Excerpt:
          </span>
          <div
            style={{
              background: "rgba(0, 0, 0, 0.4)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "var(--radius-md)",
              padding: "1rem",
              fontSize: "0.85rem",
              color: "var(--text-secondary)",
              lineHeight: 1.6,
              maxHeight: "220px",
              overflowY: "auto",
              whiteSpace: "pre-wrap",
            }}
          >
            {citation.snippet || "No textual excerpt available."}
          </div>
        </div>
      </div>
    </div>
  );
};
