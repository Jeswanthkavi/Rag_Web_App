import React, { useCallback, useEffect, useRef, useState } from "react";
import Head from "next/head";
import { ProtectedRoute } from "../components/layout/ProtectedRoute";
import { ChatSidebar } from "../components/chat/ChatSidebar";
import { ChatMessage } from "../components/chat/ChatMessage";
import { ChatInput } from "../components/chat/ChatInput";
import { CitationModal } from "../components/chat/CitationModal";
import { api } from "../lib/api";
import { APIError, Citation, Conversation, Message } from "../types";

export default function ChatPage() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoadingConversations, setIsLoadingConversations] = useState(true);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isGenerating]);

  const fetchConversations = useCallback(async () => {
    setIsLoadingConversations(true);
    try {
      const convs = await api.listConversations();
      setConversations(convs);
      if (convs.length > 0 && !activeConversationId) {
        setActiveConversationId(convs[0].id);
      }
    } catch {
      // Handled
    } finally {
      setIsLoadingConversations(false);
    }
  }, [activeConversationId]);

  useEffect(() => {
    fetchConversations();
  }, [fetchConversations]);

  const loadConversationMessages = useCallback(async (convId: string) => {
    setIsLoadingMessages(true);
    setQueryError(null);
    try {
      const detail = await api.getConversation(convId);
      setMessages(detail.messages || []);
    } catch (err: unknown) {
      const apiErr = err as APIError;
      setQueryError(apiErr.detail || "Failed to load messages.");
    } finally {
      setIsLoadingMessages(false);
    }
  }, []);

  useEffect(() => {
    if (activeConversationId) {
      loadConversationMessages(activeConversationId);
    } else {
      setMessages([]);
    }
  }, [activeConversationId, loadConversationMessages]);

  const handleNewConversation = async () => {
    try {
      const newConv = await api.createConversation("New Chat");
      setConversations((prev) => [newConv, ...prev]);
      setActiveConversationId(newConv.id);
      setMessages([]);
    } catch (err: unknown) {
      const apiErr = err as APIError;
      setQueryError(apiErr.detail || "Failed to create conversation.");
    }
  };

  const handleDeleteConversation = async (convId: string) => {
    try {
      await api.deleteConversation(convId);
      setConversations((prev) => prev.filter((c) => c.id !== convId));
      if (activeConversationId === convId) {
        const remaining = conversations.filter((c) => c.id !== convId);
        setActiveConversationId(remaining.length > 0 ? remaining[0].id : null);
      }
    } catch (err: unknown) {
      const apiErr = err as APIError;
      setQueryError(apiErr.detail || "Failed to delete conversation.");
    }
  };

  const handleSendMessage = async (queryText: string) => {
    setQueryError(null);

    // Optimistic user turn
    const tempUserMsg: Message = {
      id: `temp-${Date.now()}`,
      conversation_id: activeConversationId || "",
      user_id: "",
      role: "user",
      content: queryText,
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, tempUserMsg]);
    setIsGenerating(true);

    try {
      const response = await api.queryRAG({
        query: queryText,
        conversation_id: activeConversationId || undefined,
      });

      // If backend created a new conversation automatically or updated message ID
      if (response.conversation_id && response.conversation_id !== activeConversationId) {
        setActiveConversationId(response.conversation_id);
        fetchConversations();
      }

      const assistantMsg: Message = {
        id: response.message_id || `asst-${Date.now()}`,
        conversation_id: response.conversation_id || activeConversationId || "",
        user_id: "",
        role: "assistant",
        content: response.answer,
        citations: response.citations,
        created_at: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: unknown) {
      const apiErr = err as APIError;
      setQueryError(apiErr.detail || "Failed to generate grounded answer.");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <ProtectedRoute>
      <Head>
        <title>Chat Assistant — Advanced RAG</title>
      </Head>
      <div style={{ display: "flex", flex: 1, height: "calc(100vh - 64px)", overflow: "hidden" }}>
        {/* Sidebar */}
        <ChatSidebar
          conversations={conversations}
          activeConversationId={activeConversationId}
          isLoading={isLoadingConversations}
          onSelectConversation={setActiveConversationId}
          onNewConversation={handleNewConversation}
          onDeleteConversation={handleDeleteConversation}
        />

        {/* Main Chat Area */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", background: "var(--bg-app)" }}>
          {/* Messages Viewport */}
          <div
            style={{
              flex: 1,
              overflowY: "auto",
              padding: "2rem 1.5rem",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <div style={{ maxWidth: "800px", width: "100%", margin: "0 auto", flex: 1 }}>
              {queryError && (
                <div
                  style={{
                    background: "var(--status-error-bg)",
                    color: "var(--status-error)",
                    border: "1px solid rgba(239, 68, 68, 0.3)",
                    padding: "0.75rem 1rem",
                    borderRadius: "var(--radius-md)",
                    fontSize: "0.85rem",
                    marginBottom: "1.5rem",
                  }}
                >
                  ⚠️ {queryError}
                </div>
              )}

              {isLoadingMessages ? (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    padding: "4rem 0",
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
                      marginBottom: "1rem",
                    }}
                    className="animate-spin"
                  />
                  <p>Loading conversation history...</p>
                </div>
              ) : messages.length === 0 ? (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    textAlign: "center",
                    padding: "5rem 1rem",
                    color: "var(--text-secondary)",
                  }}
                >
                  <div
                    style={{
                      width: "60px",
                      height: "60px",
                      borderRadius: "16px",
                      background: "var(--accent-gradient)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: "1.8rem",
                      color: "#ffffff",
                      boxShadow: "var(--shadow-glow)",
                      marginBottom: "1.25rem",
                    }}
                  >
                    🧠
                  </div>
                  <h2 style={{ fontSize: "1.4rem", fontWeight: 700, marginBottom: "0.5rem" }}>
                    Ask Your Knowledge Base
                  </h2>
                  <p style={{ maxWidth: "450px", fontSize: "0.9rem", color: "var(--text-muted)", marginBottom: "2rem" }}>
                    Answers are synthesized using hybrid dense + BM25 retrieval, RRF ranking, cross-encoder reranking, and verified citation grounding.
                  </p>

                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: "0.75rem",
                      maxWidth: "580px",
                      width: "100%",
                      textAlign: "left",
                    }}
                  >
                    {[
                      "Summarize the key points in my uploaded documents",
                      "What are the return policy terms and turnaround times?",
                      "Explain the system failover and backup architecture",
                      "What is the warranty coverage period for parts?",
                    ].map((promptText, i) => (
                      <button
                        key={i}
                        onClick={() => handleSendMessage(promptText)}
                        className="btn-secondary"
                        style={{
                          padding: "0.85rem 1rem",
                          fontSize: "0.85rem",
                          textAlign: "left",
                          lineHeight: 1.4,
                        }}
                      >
                        {promptText}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                messages.map((msg) => (
                  <ChatMessage
                    key={msg.id}
                    role={msg.role}
                    content={msg.content}
                    citations={msg.citations}
                    onSelectCitation={setSelectedCitation}
                  />
                ))
              )}

              {isGenerating && (
                <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", marginBottom: "1.5rem" }}>
                  <div
                    style={{
                      width: "34px",
                      height: "34px",
                      borderRadius: "50%",
                      background: "var(--accent-gradient)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: "0.9rem",
                    }}
                  >
                    🤖
                  </div>
                  <div
                    style={{
                      background: "var(--bg-card)",
                      border: "1px solid var(--border-subtle)",
                      borderRadius: "4px 16px 16px 16px",
                      padding: "0.85rem 1.25rem",
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                      fontSize: "0.9rem",
                      color: "var(--text-accent)",
                    }}
                  >
                    <div
                      style={{
                        width: "14px",
                        height: "14px",
                        border: "2px solid rgba(99, 102, 241, 0.2)",
                        borderTopColor: "var(--accent-primary)",
                        borderRadius: "50%",
                      }}
                      className="animate-spin"
                    />
                    <span>Retrieving hybrid context & generating grounded answer...</span>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Chat Input */}
          <ChatInput
            onSendMessage={handleSendMessage}
            isLoading={isGenerating}
            disabled={isLoadingMessages}
          />
        </div>
      </div>

      {/* Citation Detail Modal */}
      <CitationModal
        citation={selectedCitation}
        onClose={() => setSelectedCitation(null)}
      />
    </ProtectedRoute>
  );
}
