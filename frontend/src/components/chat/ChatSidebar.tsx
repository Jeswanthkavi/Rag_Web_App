import React from "react";
import { Conversation } from "../../types";

interface ChatSidebarProps {
  conversations: Conversation[];
  activeConversationId: string | null;
  isLoading: boolean;
  onSelectConversation: (convId: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (convId: string) => void;
}

export const ChatSidebar: React.FC<ChatSidebarProps> = ({
  conversations,
  activeConversationId,
  isLoading,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
}) => {
  return (
    <aside
      style={{
        width: "280px",
        height: "calc(100vh - 64px)",
        background: "var(--bg-surface)",
        borderRight: "1px solid var(--border-subtle)",
        display: "flex",
        flexDirection: "column",
        flexShrink: 0,
      }}
    >
      {/* Header with New Thread Button */}
      <div style={{ padding: "1.25rem 1rem", borderBottom: "1px solid var(--border-subtle)" }}>
        <button
          onClick={onNewConversation}
          className="btn-primary"
          style={{ width: "100%", padding: "0.65rem 1rem", fontSize: "0.9rem" }}
        >
          <span>＋</span> New Chat Session
        </button>
      </div>

      {/* Conversation List */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "0.75rem 0.5rem",
          display: "flex",
          flexDirection: "column",
          gap: "0.25rem",
        }}
      >
        {isLoading && conversations.length === 0 ? (
          <div style={{ textAlign: "center", padding: "2rem 1rem", color: "var(--text-muted)", fontSize: "0.85rem" }}>
            <div
              style={{
                width: "20px",
                height: "20px",
                border: "2px solid rgba(99, 102, 241, 0.2)",
                borderTopColor: "var(--accent-primary)",
                borderRadius: "50%",
                margin: "0 auto 0.75rem",
              }}
              className="animate-spin"
            />
            Loading chats...
          </div>
        ) : conversations.length === 0 ? (
          <div
            style={{
              textAlign: "center",
              padding: "2rem 1rem",
              color: "var(--text-muted)",
              fontSize: "0.85rem",
            }}
          >
            No active conversations. Start a new chat to begin!
          </div>
        ) : (
          conversations.map((conv) => {
            const isActive = conv.id === activeConversationId;
            return (
              <div
                key={conv.id}
                onClick={() => onSelectConversation(conv.id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "0.65rem 0.75rem",
                  borderRadius: "var(--radius-md)",
                  background: isActive ? "rgba(99, 102, 241, 0.15)" : "transparent",
                  border: isActive ? "1px solid rgba(99, 102, 241, 0.3)" : "1px solid transparent",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}

                onMouseEnter={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.background = "var(--bg-card)";
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.background = "transparent";
                  }
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "0.5rem",
                    overflow: "hidden",
                    flex: 1,
                  }}
                >
                  <span style={{ fontSize: "0.9rem" }}>💬</span>
                  <span
                    style={{
                      fontSize: "0.85rem",
                      fontWeight: isActive ? 600 : 400,
                      color: isActive ? "var(--text-primary)" : "var(--text-secondary)",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {conv.title || "Untitled Chat"}
                  </span>
                </div>

                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    if (window.confirm("Delete this conversation thread?")) {
                      onDeleteConversation(conv.id);
                    }
                  }}
                  style={{
                    color: "var(--text-muted)",
                    padding: "0.2rem 0.4rem",
                    borderRadius: "4px",
                    fontSize: "0.8rem",
                    opacity: isActive ? 1 : 0.6,
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = "var(--status-error)";
                    e.currentTarget.style.background = "var(--status-error-bg)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.color = "var(--text-muted)";
                    e.currentTarget.style.background = "transparent";
                  }}
                  title="Delete Conversation"
                >
                  🗑
                </button>
              </div>
            );
          })
        )}
      </div>

      {/* Footer Info */}
      <div
        style={{
          padding: "0.85rem 1rem",
          borderTop: "1px solid var(--border-subtle)",
          fontSize: "0.75rem",
          color: "var(--text-muted)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span>⚡ Hybrid RAG Active</span>
        <span style={{ color: "var(--status-success)" }}>● Online</span>
      </div>
    </aside>
  );
};
