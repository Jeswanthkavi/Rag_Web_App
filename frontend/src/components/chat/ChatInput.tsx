import React, { useRef, useState } from "react";

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isLoading,
  disabled,
}) => {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading || disabled) return;

    onSendMessage(trimmed);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        180
      )}px`;
    }
  };

  return (
    <div
      style={{
        padding: "1rem 1.5rem",
        background: "rgba(15, 20, 31, 0.8)",
        backdropFilter: "blur(12px)",
        borderTop: "1px solid var(--border-subtle)",
        position: "relative",
      }}
    >
      <div
        style={{
          maxWidth: "800px",
          margin: "0 auto",
          position: "relative",
          display: "flex",
          alignItems: "flex-end",
          gap: "0.75rem",
          background: "var(--bg-card)",
          border: "1px solid var(--border-subtle)",
          borderRadius: "var(--radius-lg)",
          padding: "0.5rem 0.75rem 0.5rem 1rem",
          boxShadow: "var(--shadow-md)",
          transition: "border-color 0.2s ease",
        }}
      >
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your knowledge base documents..."
          rows={1}
          disabled={isLoading || disabled}
          style={{
            flex: 1,
            background: "transparent",
            border: "none",
            boxShadow: "none",
            padding: "0.4rem 0",
            resize: "none",
            maxHeight: "180px",
            fontSize: "0.95rem",
            lineHeight: 1.5,
          }}
        />

        <button
          onClick={handleSend}
          disabled={!input.trim() || isLoading || disabled}
          className="btn-primary"
          style={{
            padding: "0.5rem 1rem",
            borderRadius: "var(--radius-md)",
            fontSize: "0.9rem",
            height: "38px",
            flexShrink: 0,
          }}
        >
          {isLoading ? (
            <div
              style={{
                width: "16px",
                height: "16px",
                border: "2px solid rgba(255, 255, 255, 0.3)",
                borderTopColor: "#ffffff",
                borderRadius: "50%",
              }}
              className="animate-spin"
            />
          ) : (
            <span>Send ➔</span>
          )}
        </button>
      </div>

      <div
        style={{
          maxWidth: "800px",
          margin: "0.4rem auto 0",
          textAlign: "center",
          fontSize: "0.75rem",
          color: "var(--text-muted)",
        }}
      >
        Press <span className="code-inline">Enter</span> to send,{" "}
        <span className="code-inline">Shift+Enter</span> for new line. Grounded on your tenant documents.
      </div>
    </div>
  );
};
