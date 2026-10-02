/**
 * Centralized API Client Abstraction for Advanced RAG Platform.
 * Enforces strict typing, JWT injection, and structured error propagation.
 */

import {
  APIError,
  Conversation,
  ConversationDetail,
  DocumentItem,
  DocumentListResponse,
  RAGQueryRequest,
  RAGResponse,
  Token,
  User,
} from "../types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";

class APIClient {
  private getAuthToken(): string | null {
    if (typeof window === "undefined") return null;
    return localStorage.getItem("rag_auth_token");
  }

  public setAuthToken(token: string | null): void {
    if (typeof window === "undefined") return;
    if (token) {
      localStorage.setItem("rag_auth_token", token);
    } else {
      localStorage.removeItem("rag_auth_token");
    }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    const headers: Record<string, string> = {
      Accept: "application/json",
      ...(options.headers as Record<string, string>),
    };

    const token = this.getAuthToken();
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    if (
      options.body &&
      !(options.body instanceof FormData) &&
      !headers["Content-Type"]
    ) {
      headers["Content-Type"] = "application/json";
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      let errorMessage = `HTTP Error ${response.status}: ${response.statusText}`;
      try {
        const errorData = await response.json();
        if (typeof errorData.detail === "string") {
          errorMessage = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          errorMessage = errorData.detail
            .map((d: { msg?: string }) => d.msg || JSON.stringify(d))
            .join(", ");
        }
      } catch {
        // Response was not JSON
      }

      const apiError: APIError = {
        status: response.status,
        detail: errorMessage,
      };

      if (response.status === 401 && typeof window !== "undefined") {
        this.setAuthToken(null);
      }

      throw apiError;
    }

    if (response.status === 204) {
      return {} as T;
    }

    return response.json();
  }

  // --- Auth APIs ---

  public async register(email: string, password: string): Promise<User> {
    return this.request<User>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
  }

  public async login(email: string, password: string): Promise<Token> {
    const token = await this.request<Token>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    this.setAuthToken(token.access_token);
    return token;
  }

  public async getMe(): Promise<User> {
    return this.request<User>("/users/me");
  }

  public logout(): void {
    this.setAuthToken(null);
  }

  // --- Document APIs ---

  public async listDocuments(
    skip = 0,
    limit = 100
  ): Promise<DocumentListResponse> {
    return this.request<DocumentListResponse>(
      `/documents?skip=${skip}&limit=${limit}`
    );
  }

  public async getDocument(documentId: string): Promise<DocumentItem> {
    return this.request<DocumentItem>(`/documents/${documentId}`);
  }

  public async uploadDocument(file: File): Promise<DocumentItem> {
    const formData = new FormData();
    formData.append("file", file);

    return this.request<DocumentItem>("/documents/upload", {
      method: "POST",
      body: formData,
    });
  }

  public async deleteDocument(documentId: string): Promise<void> {
    return this.request<void>(`/documents/${documentId}`, {
      method: "DELETE",
    });
  }

  // --- Conversation APIs ---

  public async listConversations(
    skip = 0,
    limit = 50
  ): Promise<Conversation[]> {
    return this.request<Conversation[]>(
      `/conversations?skip=${skip}&limit=${limit}`
    );
  }

  public async getConversation(
    conversationId: string
  ): Promise<ConversationDetail> {
    return this.request<ConversationDetail>(
      `/conversations/${conversationId}`
    );
  }

  public async createConversation(title?: string): Promise<Conversation> {
    return this.request<Conversation>("/conversations", {
      method: "POST",
      body: JSON.stringify({ title: title || "New Conversation" }),
    });
  }

  public async updateConversation(
    conversationId: string,
    title: string
  ): Promise<Conversation> {
    return this.request<Conversation>(`/conversations/${conversationId}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    });
  }

  public async deleteConversation(conversationId: string): Promise<void> {
    return this.request<void>(`/conversations/${conversationId}`, {
      method: "DELETE",
    });
  }

  // --- RAG Query API ---

  public async queryRAG(payload: RAGQueryRequest): Promise<RAGResponse> {
    return this.request<RAGResponse>("/rag/query", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }
}

export const api = new APIClient();
