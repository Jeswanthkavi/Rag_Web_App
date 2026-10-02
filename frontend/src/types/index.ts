/**
 * TypeScript Data Models matching FastAPI Backend Schemas.
 */

export interface User {
  id: string;
  email: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
  updated_at: string;
}

export interface Token {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export type DocumentStatus = "PENDING" | "PROCESSING" | "COMPLETED" | "FAILED";

export interface DocumentItem {
  id: string;
  user_id: string;
  filename: string;
  file_type: string;
  file_size: number;
  status: DocumentStatus;
  total_chunks: number;
  error_message?: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentListResponse {
  items: DocumentItem[];
  total: number;
  skip: number;
  limit: number;
}

export interface Conversation {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface MessageCitation {
  source_id?: string;
  source_index?: number;
  document_id: string;
  chunk_id: string;
  filename: string;
  page_number?: number | null;
  chunk_index?: number;
  snippet?: string;
  verified: boolean;
}

export interface Message {
  id: string;
  conversation_id: string;
  user_id: string;
  role: "user" | "assistant";
  content: string;
  citations?: MessageCitation[] | null;
  created_at: string;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface Citation {
  source_id: string;
  source_index: number;
  document_id: string;
  chunk_id: string;
  filename: string;
  page_number?: number | null;
  chunk_index: number;
  snippet: string;
  verified: boolean;
}

export interface RAGQueryRequest {
  query: string;
  top_k?: number;
  similarity_threshold?: number;
  document_id?: string;
  conversation_id?: string;
}

export interface RAGResponse {
  query: string;
  answer: string;
  retrieved_context_count: number;
  citations: Citation[];
  conversation_id?: string | null;
  message_id?: string | null;
  model_name?: string | null;
}

export interface APIError {
  status: number;
  detail: string;
}
