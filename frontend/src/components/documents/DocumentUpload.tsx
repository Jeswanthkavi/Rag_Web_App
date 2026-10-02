import React, { useRef, useState } from "react";
import { api } from "../../lib/api";
import { APIError, DocumentItem } from "../../types";

interface DocumentUploadProps {
  onUploadSuccess: (doc: DocumentItem) => void;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({
  onUploadSuccess,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setError(null);
    setSuccessMessage(null);

    // Frontend validation
    const allowedExtensions = [".pdf", ".docx", ".txt"];
    const fileExt = `.${file.name.split(".").pop()?.toLowerCase()}`;
    if (!allowedExtensions.includes(fileExt)) {
      setError(`Unsupported file type '${fileExt}'. Please upload .pdf, .docx, or .txt.`);
      return;
    }

    const maxBytes = 25 * 1024 * 1024;
    if (file.size > maxBytes) {
      setError(`File exceeds maximum size of 25MB (${(file.size / (1024 * 1024)).toFixed(1)}MB).`);
      return;
    }

    setIsUploading(true);
    try {
      const doc = await api.uploadDocument(file);
      setSuccessMessage(`'${file.name}' uploaded successfully. Ingestion scheduled!`);
      onUploadSuccess(doc);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err: unknown) {
      const apiErr = err as APIError;
      setError(apiErr.detail || "Upload failed. Please check the file and try again.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="glass-panel" style={{ padding: "1.75rem", marginBottom: "1.5rem" }}>
      <h2 style={{ fontSize: "1.1rem", fontWeight: 600, marginBottom: "0.5rem" }}>
        Upload Knowledge Base Document
      </h2>
      <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "1.25rem" }}>
        Supported formats: PDF (with pages), DOCX, TXT. Maximum file size: 25 MB.
      </p>

      {error && (
        <div
          style={{
            background: "var(--status-error-bg)",
            color: "var(--status-error)",
            border: "1px solid rgba(239, 68, 68, 0.3)",
            padding: "0.75rem 1rem",
            borderRadius: "var(--radius-md)",
            fontSize: "0.85rem",
            marginBottom: "1rem",
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
          }}
        >
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {successMessage && (
        <div
          style={{
            background: "var(--status-success-bg)",
            color: "var(--status-success)",
            border: "1px solid rgba(16, 185, 129, 0.3)",
            padding: "0.75rem 1rem",
            borderRadius: "var(--radius-md)",
            fontSize: "0.85rem",
            marginBottom: "1rem",
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
          }}
        >
          <span>✓</span>
          <span>{successMessage}</span>
        </div>
      )}

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${
            isDragging ? "var(--accent-primary)" : "var(--border-subtle)"
          }`,
          borderRadius: "var(--radius-md)",
          padding: "2.5rem 1.5rem",
          textAlign: "center",
          background: isDragging
            ? "rgba(99, 102, 241, 0.08)"
            : "rgba(0, 0, 0, 0.2)",
          cursor: isUploading ? "not-allowed" : "pointer",
          transition: "all 0.2s ease",
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          data-testid="file-upload-input"
          accept=".pdf,.docx,.txt"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleFile(e.target.files[0]);
            }
          }}
          style={{ display: "none" }}
          disabled={isUploading}
        />


        {isUploading ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.75rem" }}>
            <div
              style={{
                width: "28px",
                height: "28px",
                border: "3px solid rgba(99, 102, 241, 0.2)",
                borderTopColor: "var(--accent-primary)",
                borderRadius: "50%",
              }}
              className="animate-spin"
            />
            <p style={{ color: "var(--text-accent)", fontWeight: 500, fontSize: "0.9rem" }}>
              Uploading and analyzing document...
            </p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.5rem" }}>
            <div
              style={{
                width: "44px",
                height: "44px",
                borderRadius: "50%",
                background: "rgba(99, 102, 241, 0.15)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "var(--accent-primary)",
                fontSize: "1.3rem",
                marginBottom: "0.25rem",
              }}
            >
              📥
            </div>
            <p style={{ fontWeight: 600, fontSize: "0.95rem" }}>
              Click to select or drag & drop a file here
            </p>
            <p style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
              PDF, DOCX, or TXT up to 25MB
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
