import React, { useCallback, useEffect, useState } from "react";
import Head from "next/head";
import { ProtectedRoute } from "../components/layout/ProtectedRoute";
import { DocumentUpload } from "../components/documents/DocumentUpload";
import { DocumentList } from "../components/documents/DocumentList";
import { api } from "../lib/api";
import { DocumentItem } from "../types";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchDocuments = useCallback(async () => {
    setIsLoading(true);
    try {
      const resp = await api.listDocuments(0, 100);
      setDocuments(resp.items);
    } catch {
      // Ignored / handled in list
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const handleUploadSuccess = (newDoc: DocumentItem) => {
    setDocuments((prev) => [newDoc, ...prev]);
  };

  const handleDeleteSuccess = (deletedId: string) => {
    setDocuments((prev) => prev.filter((d) => d.id !== deletedId));
  };

  return (
    <ProtectedRoute>
      <Head>
        <title>Knowledge Base Documents — Advanced RAG</title>
      </Head>
      <div
        style={{
          flex: 1,
          padding: "2rem",
          maxWidth: "1100px",
          width: "100%",
          margin: "0 auto",
        }}
      >
        <div style={{ marginBottom: "2rem" }}>
          <h1
            style={{
              fontSize: "1.75rem",
              fontWeight: 700,
              letterSpacing: "-0.03em",
              marginBottom: "0.5rem",
            }}
          >
            Tenant Knowledge Base
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem" }}>
            Upload and manage your private documents. Chunks and embeddings are indexed in Qdrant and BM25 for hybrid retrieval.
          </p>
        </div>

        <DocumentUpload onUploadSuccess={handleUploadSuccess} />

        <DocumentList
          documents={documents}
          isLoading={isLoading}
          onRefresh={fetchDocuments}
          onDeleteSuccess={handleDeleteSuccess}
        />
      </div>
    </ProtectedRoute>
  );
}
